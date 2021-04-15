#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <cuda.h>

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

extern "C" {
    #include "rdma.h"
    #include "timing.h"
    #include "fifo.h"
}

#define INF INT_MAX
#define THREADS_PER_BLOCK 1024

#define HUGEPAGE_SZ (4UL * 1024UL * 1024UL * 1024UL)
#define M 65536UL
#define SUB_M 4096UL
#define AGGREGATED_SZ (M * SUB_M * 8UL)

#define IO_QUEUE_SZ (HUGEPAGE_SZ / AGGREGATED_SZ)
// #define IO_QUEUE_SZ 1UL

#define NITERS 16UL // pre-check from another implementation

void print_config(struct config_t config);

struct fetch_conf {
    struct resources *res;
    uint64_t m, sub_m;
    int64_t *graph_d;
    char *hugepage_addr;
    struct fifo *sending_queue;
    struct fifo *complete_queue;
    struct timing_info *col_fetch_timing;
    struct timing_info *copy_in_timing;
};

struct request_conf {
    struct resources *res;
    uint64_t id;
    uint64_t sub_m;
};

struct fifo_entry {
    int64_t *graph_d;
};

void colstripe_reshape(double *dst, double *src, size_t m, size_t sub_m, size_t granularity) {
    uint64_t chunk, i, row, col;
    uint64_t dsize = m * sub_m;
    uint64_t multiplier = sub_m / granularity;
    double *dst_ptr, *src_ptr = src;
    for (chunk = 0; chunk < (dsize / granularity / granularity); chunk++) {
        row = chunk / multiplier;
        col = chunk % multiplier;
        dst_ptr = dst + row * sub_m * granularity + col * granularity; 
        for (i = 0; i < granularity; i++) {
            memcpy(dst_ptr, src_ptr, sizeof(double) * granularity);
            dst_ptr += sub_m;
            src_ptr += granularity;
        }
    }
}

__global__ void bellman_ford_one_iter(size_t n, size_t sub_n, size_t st, int64_t *d_subgraph, int64_t *d_dist, bool *d_has_next){
	size_t v = blockDim.x * blockIdx.x + threadIdx.x;
	size_t u;
	int64_t weight, new_dist;
	int64_t *node;
	
	if (v > sub_n) {
		return;
	}
	
	node = d_subgraph + v;
	v = v + st;
	for (u = 0; u < n; u++){
		weight = node[u * sub_n]; // row is src, col is dst
		if (weight > 0) {
			new_dist = d_dist[u] + weight;
			if(new_dist < d_dist[v]){
				d_dist[v] = new_dist;
				*d_has_next = true;
			}
		}
	}
}

int cudaMemcpyFromMmap(struct fetch_conf *conf, uint64_t id, uint64_t st, uint64_t en, uint64_t size, uint64_t op, uint64_t which,
    char *dst, const char *src, const size_t length, struct timing_info *fetch_timing) {
    struct response *res = NULL;

    timing_info_push_start(fetch_timing);
    sock_write_request(conf->res->req_sock, id, st, en, size, op, which);
    sock_read_data(conf->res->req_sock);

    res = sock_read_offset(conf->res->sock);
    if (res == NULL) {
        fprintf(stderr, "sync error before RDMA ops\n");
        return 1;
    }

    // if (res->id == 0) {
    //     printf("fetching row [%lu:%lu]\n", res->x, res->y);
    // } else {
    //     printf("fetching col [%lu:%lu]\n", res->x, res->y);
    // }
    // printf("offset: %lu\n", res->offset);

    timing_info_push_end(fetch_timing);

    timing_info_push_start(conf->copy_in_timing);
    cudaMemcpy(dst, src + res->offset, length, cudaMemcpyHostToDevice);
    timing_info_push_end(conf->copy_in_timing);

    free(res);
    if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
        fprintf(stderr, "sync error before RDMA ops\n");
        return 1;
    }
    return 0;
}

void *request_thread(void *args) {
    struct request_conf *conf = (struct request_conf *) args;
    uint64_t st;

    for (st = 0; st < M / SUB_M; st++) {
        sock_write_request(conf->res->req_sock, conf->id, st, st+1, conf->sub_m, 3, 0);
        sock_read_data(conf->res->req_sock);
    }
    return NULL;
}

int nds_bfs(struct resources *res, uint64_t id, uint64_t num_of_vertices, uint64_t num_of_subvertices) {
    size_t stripe_size, st, i;
    int64_t *graph_d;

    // result
    int64_t *dist, *d_dist;
    bool *d_has_next, h_has_next;

    size_t total_iteration;

	uint64_t iter = 0;

    struct timing_info *col_fetch_timing;
    struct timing_info *reshape_timing;    
    struct timing_info *copy_in_timing;
    struct timing_info *kernel_timing;
    struct timing_info *copy_out_timing;    
    
    struct fetch_conf f_conf;

    struct timeval h_start, h_end;
    long duration;

    struct response *resp = NULL;
    double *reshaped_data = (double *) calloc(M * SUB_M, sizeof(double));

    // initialization
    total_iteration = NITERS * (num_of_vertices / num_of_subvertices);

    col_fetch_timing = timing_info_new(total_iteration);
    if (col_fetch_timing == NULL) {
        printf("cannot create col_fetch_timing\n");
        return -1;
    }
    
    reshape_timing = timing_info_new(total_iteration);
    if (reshape_timing == NULL) {
        printf("cannot create reshape_timing\n");
        return -1;
    }

    copy_in_timing = timing_info_new(total_iteration * 2);
    if (copy_in_timing == NULL) {
        printf("cannot create copy_in_timing\n");
        return -1;
    }

    kernel_timing = timing_info_new(total_iteration);
    if (kernel_timing == NULL) {
        printf("cannot create kernel_timing\n");
        return -1;
    }

    copy_out_timing = timing_info_new(total_iteration);
    if (copy_out_timing == NULL) {
        printf("cannot create copy_out_timing\n");
        return -1;
    }

    // subgraph initialization
    stripe_size = num_of_vertices * num_of_subvertices * sizeof(int64_t);
    cudaMalloc((void **) &graph_d, stripe_size);

    // Bellman-Ford initialization
	cudaMalloc(&d_dist, sizeof(int64_t) * num_of_vertices);
	cudaMalloc(&d_has_next, sizeof(bool));

    bool has_negative_cycle = false;
    
    dist = (int64_t *) calloc(sizeof(int64_t), num_of_vertices);
	for(i = 0 ; i < num_of_vertices; i++){
		dist[i] = INF;
	}

	dist[0] = 0;
    cudaMemcpy(d_dist, dist, sizeof(int64_t) * num_of_vertices, cudaMemcpyHostToDevice);

    // create thread here
    f_conf.res = res;
    f_conf.m = num_of_vertices;
    f_conf.sub_m = num_of_subvertices;
    f_conf.graph_d = graph_d;
    f_conf.hugepage_addr = res->buf;
    f_conf.col_fetch_timing = col_fetch_timing;
    f_conf.copy_in_timing = copy_in_timing;

    timing_info_set_starting_time(col_fetch_timing);
    timing_info_set_starting_time(reshape_timing);
    timing_info_set_starting_time(copy_in_timing);
    timing_info_set_starting_time(kernel_timing);
    timing_info_set_starting_time(copy_out_timing);

    gettimeofday(&h_start, NULL);
    // blockGEMM
    do {
        printf("iter: %lu\n", iter);
        //if no thread changes this value then the loop stops
		h_has_next = false;
        cudaMemset(d_has_next, 0, sizeof(bool));
        
        for (st = 0; st < num_of_vertices / num_of_subvertices; st++) {
            timing_info_push_start(col_fetch_timing);
            sock_write_request(res->req_sock, id, st, st+1, SUB_M, 3, 0);
            sock_read_data(res->req_sock);
        
            resp = sock_read_offset(res->sock);
            if (resp == NULL) {
                fprintf(stderr, "sync error before RDMA ops\n");
                return -1;
            }

            timing_info_push_end(col_fetch_timing);

            timing_info_push_start(reshape_timing);
            colstripe_reshape(reshaped_data, (double *) (res->buf + resp->offset), M, SUB_M, 256UL);
            if (sock_write_data(res->sock)) { /* just send a dummy char back and forth */
                fprintf(stderr, "sync error before RDMA ops\n");
                return -1;
            }
            free(resp);
            timing_info_push_end(reshape_timing);

            timing_info_push_start(copy_in_timing);
            cudaMemcpy(graph_d, reshaped_data, M * SUB_M * sizeof(double), cudaMemcpyHostToDevice);
            timing_info_push_end(copy_in_timing);

            timing_info_push_start(kernel_timing);
			bellman_ford_one_iter<<<(num_of_subvertices+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(num_of_vertices, num_of_subvertices, st * num_of_subvertices, graph_d, d_dist, d_has_next);
            timing_info_push_end(kernel_timing);
        }
        
        timing_info_push_start(copy_out_timing);
		cudaMemcpy(&h_has_next, d_has_next, sizeof(bool), cudaMemcpyDeviceToHost);
        timing_info_push_end(copy_out_timing);

        iter++;
        if (iter >= num_of_vertices - 1){
			has_negative_cycle = true;
			break;
		}
    } while (h_has_next);

    // send a signal to tell storage backend the iteration is done.
    sock_write_request(res->req_sock, -1, st, st+1, SUB_M, 0, 0);
    sock_read_data(res->req_sock);
    
    if (!has_negative_cycle){
        cudaMemcpy(dist, d_dist, sizeof(int64_t) * num_of_vertices, cudaMemcpyDeviceToHost);
	} 
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    
    printf("Bellman-Ford End-to-end duration: %f ms\n", (float) duration / 1000);    
	printf("Kernel Executed %lu times\n", iter);

    printf("Col fetch time: %f ms\n", (float) timing_info_duration(col_fetch_timing) / 1000);
    printf("Reshape time: %f ms\n", (float) timing_info_duration(reshape_timing) / 1000);
    printf("Copy in time: %f ms\n", (float) timing_info_duration(copy_in_timing) / 1000);
    printf("Kernel time: %f ms\n", (float) timing_info_duration(kernel_timing) / 1000);
    printf("copy out time: %f ms\n", (float) timing_info_duration(copy_out_timing) / 1000);
    
    struct timestamps *tss = NULL;
    FILE *fptr;
    tss = timing_info_get_timestamps(col_fetch_timing);
    fptr = fopen("col_fetch_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(col_fetch_timing);

    tss = timing_info_get_timestamps(reshape_timing);
    fptr = fopen("reshape_timing_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(reshape_timing);

    tss = timing_info_get_timestamps(copy_in_timing);
    fptr = fopen("copy_in_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(copy_in_timing);

    tss = timing_info_get_timestamps(kernel_timing);
    fptr = fopen("kernel_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(kernel_timing);

    tss = timing_info_get_timestamps(copy_out_timing);
    fptr = fopen("copy_out_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(copy_out_timing);

    
    fptr = fopen("log.txt", "w");
    if (!has_negative_cycle){
        for (i = 0; i < num_of_vertices; i++) {
			if (dist[i] > INF) {
				dist[i] = INF;
			}
			fprintf(fptr, "%lu %lu\n", i, dist[i]);
        }
	} else {
		fprintf(fptr, "FOUND NEGATIVE CYCLE!\n");
	}
    fclose(fptr);

	// cleanup
    free(dist);
    free(reshaped_data);
    cudaFree(graph_d);
	cudaFree(d_dist);
    cudaFree(d_has_next);
    return 0;
}

/******************************************************************************
 * Function: print_config
 *
 * Input
 * none
 *
 * Output
 * none
 *
 * Returns
 * none
 *
 * Description
 * Print out config information
 ******************************************************************************/
void print_config(struct config_t config) {
    fprintf(stdout, " ------------------------------------------------\n");
    fprintf(stdout, " Device name : \"%s\"\n", config.dev_name);
    fprintf(stdout, " IB port : %u\n", config.ib_port);
    if (config.server_name)
        fprintf(stdout, " IP : %s\n", config.server_name);
    fprintf(stdout, " TCP port : %u\n", config.tcp_port);
    if (config.gid_idx >= 0)
        fprintf(stdout, " GID index : %u\n", config.gid_idx);
    fprintf(stdout, " ------------------------------------------------\n\n");
}

int main(int argc, char *argv[]) {
    int rc = 0;
    uint64_t matrix_id, n, sub_n;

    int hugepage_fd;
    char *hugepage_addr;

    // RDMA
    struct resources res;
    struct config_t config = {
        "mlx4_0",  /* dev_name */
        NULL,  /* server_name */
        19875, /* tcp_port */
        1,     /* ib_port */
        0     /* gid_idx */
    };

    // default the iteration is 4 times
    if (argc < 5) {
        printf("usage: %s <matrix_id> <# of vertices> <# of subvertices> <port>\n", argv[0]);
        exit(1);
    } 
    matrix_id = (uint64_t) atoll(argv[1]);
    n = (uint64_t) atoll(argv[2]);
    sub_n = (uint64_t) atoll(argv[3]);
    config.tcp_port = atoi(argv[4]);

    /* print the used parameters for info*/
    print_config(config);
    
    printf("mapping hugepage\n");
    hugepage_fd = open("/dev/hugepages/tensorstore", O_RDWR, 0755);
    if (hugepage_fd < 0) {
        perror("open");
        exit(1);
    }

    hugepage_addr = (char *) mmap(0, BUF_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, hugepage_fd, 0);
    if (hugepage_addr==MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    res.buf = hugepage_addr;
    memset(hugepage_addr, 0, BUF_SIZE);

    printf("hugepage starting address is: %p\n", hugepage_addr);
    printf("socket connection\n");
    rc = make_two_tcp_connection(&res, &config);
    if (rc < 0) {
        perror("sock connect");
        exit(1);
    }

    printf("calculating the result of pagerank\n");
    rc = nds_bfs(&res, matrix_id, n, sub_n);
    
    close(res.sock);
    close(res.req_sock);
    munmap(hugepage_addr, BUF_SIZE);
    close(hugepage_fd);
    return rc;
}
