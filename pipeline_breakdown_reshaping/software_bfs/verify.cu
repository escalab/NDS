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
    uint64_t *graph_d;
    bool *graph_mask_h;
    char *hugepage_addr;
    struct fifo *sending_queue;
    struct fifo *complete_queue;
    struct timing_info *row_fetch_timing;
    struct timing_info *copy_in_timing;
};

struct request_conf {
    struct resources *res;
    bool *graph_mask_h;
    uint64_t id;
    uint64_t sub_m;
};

struct fifo_entry {
    uint64_t *graph_d;
};

void rowstripe_reshape(double *dst, double *src, size_t m, size_t sub_m, size_t granularity) {
    uint64_t chunk, i, row, col;
    uint64_t dsize = m * sub_m;
    uint64_t multiplier = m / granularity;
    double *dst_ptr, *src_ptr = src;
    for (chunk = 0; chunk < (dsize / granularity / granularity); chunk++) {
        row = chunk / multiplier;
        col = chunk % multiplier;
        dst_ptr = dst + row * m * granularity + col * granularity; 
        for (i = 0; i < granularity; i++) {
            memcpy(dst_ptr, src_ptr, sizeof(double) * granularity);
            dst_ptr += m;
            src_ptr += granularity;
        }
    }
}

__global__ void
Kernel2(bool* g_graph_mask, bool *g_updating_graph_mask, bool* g_graph_visited, bool *g_over, int no_of_nodes)
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < no_of_nodes && g_updating_graph_mask[tid])
	{
		g_graph_mask[tid] = true;
		g_graph_visited[tid] = true;
		*g_over = true;
		g_updating_graph_mask[tid] = false;
	}
}

__global__ void
Kernel(size_t st, uint64_t* g_graph_nodes, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int no_of_nodes, int num_of_subvertices) 
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t node_id = tid + st * num_of_subvertices;
	if(node_id < no_of_nodes && g_graph_mask[node_id])
	{
		g_graph_mask[node_id] = false;
		uint64_t offset = tid * no_of_nodes;
		uint64_t *node = g_graph_nodes + offset;
		for(int i = 0; i < no_of_nodes; i++)
		{
			if (node[i]) {
				if(!g_graph_visited[i])
				{
					g_cost[i] = g_cost[node_id] + 1;
					g_updating_graph_mask[i] = true;
				}
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

int nds_bfs(struct resources *res, uint64_t id, uint64_t num_of_vertices, uint64_t num_of_subvertices) {
    size_t stripe_size, st, i;
    uint64_t *graph_d;

    int source = 0;

    // result
    int *cost_h, *cost_d;
    bool *graph_mask_h, *graph_mask_d, *updating_graph_mask_d, *graph_visited_d;
        
    size_t total_iteration;
    bool *d_over;
	bool h_over;
	uint64_t iter = 0, fetch_iter = 0;

    struct timing_info *row_fetch_timing;
    struct timing_info *reshape_timing;    
    struct timing_info *copy_in_timing;
    struct timing_info *bfs_1_timing;
    struct timing_info *bfs_2_timing;
    struct timing_info *copy_out_timing;    
    
    struct fetch_conf f_conf;

    struct timeval h_start, h_end;
    long duration;

    struct response *resp = NULL;
    double *reshaped_data = (double *) calloc(M * SUB_M, sizeof(double));

    // initialization
    total_iteration = NITERS * (num_of_vertices / num_of_subvertices);

    row_fetch_timing = timing_info_new(total_iteration);
    if (row_fetch_timing == NULL) {
        printf("cannot create row_fetch_timing\n");
        return -1;
    }

    reshape_timing = timing_info_new(total_iteration);
    if (reshape_timing == NULL) {
        printf("cannot create reshape_timing\n");
        return -1;
    }

    copy_in_timing = timing_info_new(total_iteration);
    if (copy_in_timing == NULL) {
        printf("cannot create copy_in_timing\n");
        return -1;
    }

    bfs_1_timing = timing_info_new(total_iteration);
    if (bfs_1_timing == NULL) {
        printf("cannot create bfs_1_timing\n");
        return -1;
    }

    bfs_2_timing = timing_info_new(NITERS);
    if (bfs_2_timing == NULL) {
        printf("cannot create bfs_1_timing\n");
        return -1;
    }

    copy_out_timing = timing_info_new(total_iteration);
    if (copy_out_timing == NULL) {
        printf("cannot create copy_out_timing\n");
        return -1;
    }

    // subgraph initialization
    stripe_size = num_of_vertices * num_of_subvertices * sizeof(uint64_t);
    cudaMalloc((void **) &graph_d, stripe_size);

	// BFS initialization
	graph_mask_h = (bool *) calloc(num_of_vertices, sizeof(bool));
	cudaMalloc((void **) &graph_mask_d, sizeof(bool) * num_of_vertices);
	graph_mask_h[source] = true;
	cudaMemcpy(graph_mask_d, graph_mask_h, sizeof(bool) * num_of_vertices, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &updating_graph_mask_d, sizeof(bool) * num_of_vertices);
	cudaMemset(updating_graph_mask_d, 0, sizeof(bool) * num_of_vertices);

	cudaMalloc((void **) &graph_visited_d, sizeof(bool) * num_of_vertices);
	cudaMemset(graph_visited_d + source, true, sizeof(bool));

	cost_h = (int *) malloc(sizeof(int) * num_of_vertices);
	cudaMalloc((void **) &cost_d, sizeof(int) * num_of_vertices);
	for (i = 0; i < num_of_vertices; i++) {
		cost_h[i] = -1;
	}	
	cost_h[source] = 0;
	cudaMemcpy(cost_d, cost_h, sizeof(int) * num_of_vertices, cudaMemcpyHostToDevice);

	//make a bool to check if the execution is over
	cudaMalloc((void**) &d_over, sizeof(bool));

    // create thread here
    f_conf.res = res;
    f_conf.m = num_of_vertices;
    f_conf.sub_m = num_of_subvertices;
    f_conf.graph_d = graph_d;
    f_conf.hugepage_addr = res->buf;
    f_conf.row_fetch_timing = row_fetch_timing;
    f_conf.copy_in_timing = copy_in_timing;
    f_conf.graph_mask_h = graph_mask_h;

    timing_info_set_starting_time(row_fetch_timing);
    timing_info_set_starting_time(reshape_timing);
    timing_info_set_starting_time(copy_in_timing);
    timing_info_set_starting_time(bfs_1_timing);
    timing_info_set_starting_time(bfs_2_timing);
    timing_info_set_starting_time(copy_out_timing);

    gettimeofday(&h_start, NULL);
    // blockGEMM
    do {
        printf("iter: %lu\n", iter);
        //if no thread changes this value then the loop stops
		h_over = false;
        cudaMemset(d_over, 0, sizeof(bool));

        for (st = 0; st < num_of_vertices / num_of_subvertices; st++) {
            for (i = st * num_of_subvertices; i < (st+1) * num_of_subvertices; i++) {
                if (graph_mask_h[i]) {
                    // printf("st: %lu\n", st * num_of_subvertices);
                    timing_info_push_start(row_fetch_timing);
                    sock_write_request(res->req_sock, id, st, st+1, SUB_M, 2, 0);
                    sock_read_data(res->req_sock);
                
                    resp = sock_read_offset(res->sock);
                    if (resp == NULL) {
                        fprintf(stderr, "sync error before RDMA ops\n");
                        return -1;
                    }
        
                    timing_info_push_end(row_fetch_timing);
        
                    timing_info_push_start(reshape_timing);
                    rowstripe_reshape(reshaped_data, (double *) (res->buf + resp->offset), M, SUB_M, 256UL);
                    if (sock_write_data(res->sock)) { /* just send a dummy char back and forth */
                        fprintf(stderr, "sync error before RDMA ops\n");
                        return -1;
                    }
                    free(resp);
                    timing_info_push_end(reshape_timing);

                    timing_info_push_start(copy_in_timing);
                    cudaMemcpy(graph_d, reshaped_data, M * SUB_M * sizeof(double), cudaMemcpyHostToDevice);
                    timing_info_push_end(copy_in_timing);
   
                    fetch_iter++;

                    timing_info_push_start(bfs_1_timing);
                    Kernel<<<(num_of_subvertices+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(st, graph_d, graph_mask_d, updating_graph_mask_d, graph_visited_d, cost_d, num_of_vertices, num_of_subvertices);
                    timing_info_push_end(bfs_1_timing);
                    break;
                }
            }
        }

        // expanding the search to the next step
        timing_info_push_start(bfs_2_timing);
		Kernel2<<<(num_of_vertices+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(graph_mask_d, updating_graph_mask_d, graph_visited_d, d_over, num_of_vertices);
        timing_info_push_end(bfs_2_timing);
        
        timing_info_push_start(copy_out_timing);
		cudaMemcpy(&h_over, d_over, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaMemcpy(graph_mask_h, graph_mask_d, sizeof(bool) * num_of_vertices, cudaMemcpyDeviceToHost);
        timing_info_push_end(copy_out_timing);
        iter++;
    }
    while (h_over);

    // send a signal to tell storage backend the iteration is done.
    sock_write_request(res->req_sock, -1, st, st+1, SUB_M, 0, 0);
    sock_read_data(res->req_sock);

	cudaMemcpy(cost_h, cost_d, sizeof(int) * num_of_vertices, cudaMemcpyDeviceToHost);

    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    
    printf("BFS End-to-end duration: %f ms\n", (float) duration / 1000);    
	printf("Kernel Executed %lu times, Row fetched %lu times\n", iter, fetch_iter);

    printf("Row fetch time: %f ms\n", (float) timing_info_duration(row_fetch_timing) / 1000);
    printf("Reshape time: %f ms\n", (float) timing_info_duration(reshape_timing) / 1000);
    printf("Copy in time: %f ms\n", (float) timing_info_duration(copy_in_timing) / 1000);
    printf("Kernel 1 time: %f ms\n", (float) timing_info_duration(bfs_1_timing) / 1000);
    printf("Kernel 2 time: %f ms\n", (float) timing_info_duration(bfs_2_timing) / 1000);
    printf("copy out time: %f ms\n", (float) timing_info_duration(copy_out_timing) / 1000);
    
    struct timestamps *tss = NULL;
    FILE *fptr;
    tss = timing_info_get_timestamps(row_fetch_timing);
    fptr = fopen("row_fetch_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(row_fetch_timing);

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

    tss = timing_info_get_timestamps(bfs_1_timing);
    fptr = fopen("bfs_1_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(bfs_1_timing);

    tss = timing_info_get_timestamps(bfs_2_timing);
    fptr = fopen("bfs_2_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(bfs_2_timing);

    tss = timing_info_get_timestamps(copy_out_timing);
    fptr = fopen("copy_out_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(copy_out_timing);

    fptr = fopen("log.txt", "w");
    for (i = 0; i < num_of_vertices; i++) {
        fprintf(fptr, "%lu) cost:%d\n", i, cost_h[i]);
    }
    fclose(fptr);

	// cleanup
	free(cost_h);
	free(graph_mask_h);
    free(reshaped_data);
    cudaFree(graph_d);
    cudaFree(graph_mask_d);
    cudaFree(updating_graph_mask_d);
    cudaFree(graph_visited_d);
	cudaFree(cost_d);
	cudaFree(d_over);
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
