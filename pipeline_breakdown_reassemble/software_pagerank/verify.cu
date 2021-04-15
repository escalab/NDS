extern "C" {
    #include "rdma.h"
    #include "timing.h"
    #include "fifo.h"
}

#define RANDOMRESETPROB 0.15
#define NUM_THREADS 1024

#define HUGEPAGE_SZ (4UL * 1024UL * 1024UL * 1024UL)
#define M 65536UL
#define SUB_M 4096UL
#define AGGREGATED_SZ (M * SUB_M * 8UL)

#define IO_QUEUE_SZ (HUGEPAGE_SZ / AGGREGATED_SZ / 2UL)
// #define IO_QUEUE_SZ 1UL

#define NITERS 4UL

void print_config(struct config_t config);

struct fetch_conf {
    struct resources *res;
    uint64_t m, sub_m;
    int64_t *outedges, *inedges;
    char *hugepage_addr;
    struct fifo *sending_queue;
    struct fifo *complete_queue;
    struct timing_info *row_fetch_timing;
    struct timing_info *col_fetch_timing;
    struct timing_info *copy_in_timing;
};

struct request_conf {
    struct resources *res;
    uint64_t id;
    uint64_t sub_m;
};

struct fifo_entry {
    int64_t *outedges, *inedges;
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

__global__ void pagerank_update(double* prev_pr, double* curr_pr, double *vertices, size_t st, int64_t* inedges, int64_t *outedges, int m, int sub_m, int iter, int niters) {
    // v.outc is num_outedges()
    // needs: v.num_inedges(), v.inedge(), v.id(), v.outc, v.set_data
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (id >= sub_m) {
        return;
    }

    int64_t *outedge = outedges + id * m;
    int64_t *inedge = inedges + id;
    int i, outc = 0;
    double sum = 0;

    id = st + id;
    for (i = 0; i < m; i++) {
        if (i != id && outedge[i] != 0) {
            outc++;
        }
    }

    // first iteration
    if (iter > 0) {
        for (i = 0; i < m; i++) {
            // we don't consider self-loop
            if (inedge[i * sub_m] && i != id) {
                sum += prev_pr[i];
            }
        }
        if (outc > 0) {
            curr_pr[id] = (RANDOMRESETPROB + (1 - RANDOMRESETPROB) * sum) / (double) outc;
        } else {
            curr_pr[id] = (RANDOMRESETPROB + (1 - RANDOMRESETPROB) * sum);
        }
    } else if (iter == 0) {
        if (outc > 0) {
            curr_pr[id] = 1.0f / (double) outc;
        }
    }

    // for the last iteration
    if (iter == niters - 1) {
        if (outc > 0) {
            vertices[id] = curr_pr[id] * (double) outc;
        } else {
            vertices[id] = curr_pr[id];
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
    uint64_t i, st;

    for (i = 0; i < NITERS; i++) {
        for (st = 0; st < M / SUB_M; st++) {
            sock_write_request(conf->res->req_sock, conf->id, st, st+1, SUB_M, 2, 0);
            sock_read_data(conf->res->req_sock);

            sock_write_request(conf->res->req_sock, conf->id, st, st+1, SUB_M, 3, 1);
            sock_read_data(conf->res->req_sock);
        }
    }
    sock_write_request(conf->res->req_sock, -1, st, st+1, SUB_M, 0, 0);
    sock_read_data(conf->res->req_sock);
    return NULL;
}

int nds_pagerank(struct resources *res, uint64_t id, uint64_t m, uint64_t sub_m) {
    size_t i, st;    
    int64_t *outedges, *inedges;

    // result
    double *vertices;
    double *prev_pr_d, *curr_pr_d, *vertices_d;
    
    size_t total_iteration;
    uint64_t stripe_size;

    struct timing_info *row_fetch_timing;
    struct timing_info *row_reshape_timing;    
    struct timing_info *col_fetch_timing;
    struct timing_info *col_reshape_timing;    
    struct timing_info *copy_in_timing;
    struct timing_info *pagerank_timing;
    struct timing_info *copy_out_timing;    
    
    struct fetch_conf f_conf;

    struct timeval h_start, h_end;
    long duration;

    struct response *resp = NULL;
    double *reshaped_data = (double *) calloc(M * SUB_M, sizeof(double));

    // initialization
    total_iteration = NITERS * (m / sub_m);

    row_fetch_timing = timing_info_new(total_iteration);
    if (row_fetch_timing == NULL) {
        printf("cannot create row_fetch_timing\n");
        return -1;
    }

    row_reshape_timing = timing_info_new(total_iteration);
    if (row_reshape_timing == NULL) {
        printf("cannot create row_reshape_timing\n");
        return -1;
    }

    col_fetch_timing = timing_info_new(total_iteration);
    if (col_fetch_timing == NULL) {
        printf("cannot create col_fetch_timing\n");
        return -1;
    }

    col_reshape_timing = timing_info_new(total_iteration);
    if (col_reshape_timing == NULL) {
        printf("cannot create col_reshape_timing\n");
        return -1;
    }

    copy_in_timing = timing_info_new(total_iteration * 2);
    if (copy_in_timing == NULL) {
        printf("cannot create copy_in_timing\n");
        return -1;
    }

    pagerank_timing = timing_info_new(total_iteration);
    if (pagerank_timing == NULL) {
        printf("cannot create pagerank_timing\n");
        return -1;
    }

    copy_out_timing = timing_info_new(total_iteration);
    if (copy_out_timing == NULL) {
        printf("cannot create copy_out_timing\n");
        return -1;
    }

    // subgraph initialization
    stripe_size = m * sub_m * sizeof(int64_t);
    cudaMalloc((void **) &outedges, stripe_size);
    cudaMalloc((void **) &inedges, stripe_size);

    // PR initialization
    vertices = (double *) malloc(sizeof(double) * m);
    for (i = 0; i < m; i++) {
        vertices[i] = RANDOMRESETPROB;
    }

    // malloc
    cudaMalloc((void **) &vertices_d, sizeof(double) * m);
    cudaMalloc((void **) &prev_pr_d, sizeof(double) * m);
    cudaMalloc((void **) &curr_pr_d, sizeof(double) * m);

    // setup memory
    cudaMemset(vertices_d, 0, sizeof(double) * m);
    cudaMemcpy(prev_pr_d, vertices, sizeof(double) * m, cudaMemcpyHostToDevice);
    cudaMemset(curr_pr_d, 0, sizeof(double) * m);
    memset(vertices, 0, sizeof(double) * m);

    // create thread here
    f_conf.res = res;
    f_conf.m = m;
    f_conf.sub_m = sub_m;
    f_conf.outedges = outedges;
    f_conf.inedges = inedges;
    f_conf.hugepage_addr = res->buf;
    f_conf.row_fetch_timing = row_fetch_timing;
    f_conf.col_fetch_timing = col_fetch_timing;
    f_conf.copy_in_timing = copy_in_timing;

    timing_info_set_starting_time(row_fetch_timing);
    timing_info_set_starting_time(row_reshape_timing);
    timing_info_set_starting_time(col_fetch_timing);
    timing_info_set_starting_time(col_reshape_timing);
    timing_info_set_starting_time(copy_in_timing);
    timing_info_set_starting_time(pagerank_timing);
    timing_info_set_starting_time(copy_out_timing);

    gettimeofday(&h_start, NULL);
    // blockGEMM
    for (i = 0; i < NITERS; i++) {
        printf("iter: %lu\n", i);
        for (st = 0; st < m / sub_m; st++) {
            printf("st: %lu\n", st * sub_m);
            // outedges
            timing_info_push_start(row_fetch_timing);
            sock_write_request(res->req_sock, id, st, st+1, SUB_M, 2, 0);
            sock_read_data(res->req_sock);
        
            resp = sock_read_offset(res->sock);
            if (resp == NULL) {
                fprintf(stderr, "sync error before RDMA ops\n");
                return -1;
            }

            timing_info_push_end(row_fetch_timing);

            timing_info_push_start(row_reshape_timing);
            rowstripe_reshape(reshaped_data, (double *) (res->buf + resp->offset), M, SUB_M, 256UL);
            if (sock_write_data(res->sock)) { /* just send a dummy char back and forth */
                fprintf(stderr, "sync error before RDMA ops\n");
                return -1;
            }
            free(resp);
            timing_info_push_end(row_reshape_timing);

            timing_info_push_start(copy_in_timing);
            cudaMemcpy(outedges, reshaped_data, M * SUB_M * sizeof(double), cudaMemcpyHostToDevice);
            timing_info_push_end(copy_in_timing);

            // inedges
            timing_info_push_start(col_fetch_timing);
            sock_write_request(res->req_sock, id, st, st+1, SUB_M, 3, 0);
            sock_read_data(res->req_sock);
        
            resp = sock_read_offset(res->sock);
            if (resp == NULL) {
                fprintf(stderr, "sync error before RDMA ops\n");
                return -1;
            }

            timing_info_push_end(col_fetch_timing);

            timing_info_push_start(col_reshape_timing);
            colstripe_reshape(reshaped_data, (double *) (res->buf + resp->offset), M, SUB_M, 256UL);
            if (sock_write_data(res->sock)) { /* just send a dummy char back and forth */
                fprintf(stderr, "sync error before RDMA ops\n");
                return -1;
            }
            free(resp);
            timing_info_push_end(col_reshape_timing);

            timing_info_push_start(copy_in_timing);
            cudaMemcpy(inedges, reshaped_data, M * SUB_M * sizeof(double), cudaMemcpyHostToDevice);
            timing_info_push_end(copy_in_timing);
    
            timing_info_push_start(pagerank_timing);
            pagerank_update<<<(sub_m+NUM_THREADS-1)/NUM_THREADS, NUM_THREADS>>>(prev_pr_d, curr_pr_d, vertices_d, st * sub_m, inedges, outedges, m, sub_m, i, NITERS);
            timing_info_push_end(pagerank_timing);
        }

        timing_info_push_start(copy_out_timing);
        cudaMemcpy(prev_pr_d, curr_pr_d, sizeof(double) * m, cudaMemcpyDeviceToDevice);
        timing_info_push_end(copy_out_timing);
    }
    cudaMemcpy(vertices, vertices_d, sizeof(double) * m, cudaMemcpyDeviceToHost);

    sock_write_request(res->req_sock, -1, st, st+1, SUB_M, 0, 0);
    sock_read_data(res->req_sock);

    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("GEMM duration: %f ms\n", (float) duration / 1000);    

    printf("Row fetch time: %f ms\n", (float) timing_info_duration(row_fetch_timing) / 1000);
    printf("Row reshape time: %f ms\n", (float) timing_info_duration(row_reshape_timing) / 1000);

    printf("Col fetch time: %f ms\n", (float) timing_info_duration(col_fetch_timing) / 1000);
    printf("Col reshape time: %f ms\n", (float) timing_info_duration(col_reshape_timing) / 1000);

    printf("Copy in time: %f ms\n", (float) timing_info_duration(copy_in_timing) / 1000);
    printf("GEMM time: %f ms\n", (float) timing_info_duration(pagerank_timing) / 1000);
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

    tss = timing_info_get_timestamps(row_reshape_timing);
    fptr = fopen("row_reshape_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(row_reshape_timing);

    tss = timing_info_get_timestamps(col_fetch_timing);
    fptr = fopen("col_fetch_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(col_fetch_timing);

    tss = timing_info_get_timestamps(col_reshape_timing);
    fptr = fopen("col_reshape_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(col_reshape_timing);

    tss = timing_info_get_timestamps(copy_in_timing);
    fptr = fopen("copy_in_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(copy_in_timing);

    tss = timing_info_get_timestamps(pagerank_timing);
    fptr = fopen("gemm_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(pagerank_timing);

    tss = timing_info_get_timestamps(copy_out_timing);
    fptr = fopen("copy_out_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(copy_out_timing);

    fptr = fopen("log.txt", "w");
    for (i = 0; i < m; i++) {
        fprintf(fptr, "%lu %f\n", i, vertices[i]);
    }
    fclose(fptr);
    free(reshaped_data);
    cudaFree(outedges);
    cudaFree(inedges);
    cudaFree(vertices_d);
    cudaFree(prev_pr_d);
    cudaFree(curr_pr_d);
    free(vertices);
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
    rc = nds_pagerank(&res, matrix_id, n, sub_n);
    
    close(res.sock);
    close(res.req_sock);
    munmap(hugepage_addr, BUF_SIZE);
    close(hugepage_fd);
    return rc;
}
