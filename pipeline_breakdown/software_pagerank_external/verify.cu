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

#define IO_QUEUE_SZ (HUGEPAGE_SZ / AGGREGATED_SZ)
// #define IO_QUEUE_SZ 1UL

#define NITERS 4UL

void print_config(struct config_t config);

struct fetch_conf {
    struct resources *res;
    uint64_t m, sub_m;
    double *outedges, *inedges;
    char *hugepage_addr;
    struct fifo *fetching_queue;
    struct fifo *kernel_queue;
    struct timing_info *row_fetch_timing;
    struct timing_info *col_fetch_timing;
    struct timing_info *copy_in_timing;
};

struct request_conf {
    struct resources *res;
    struct fifo *request_queue;
    struct fifo *fetching_queue;
};

struct fifo_entry {
    int id;
    int op;
    int which;
    uint64_t sub_m;
    size_t st;
    double *edges;
};

// construct vertices metadata first
__global__ void pagerank_update(double *vertices, size_t st, double *in_graph, double *out_graph, int *updated, size_t num_of_vertices, size_t num_of_subvertices, int iter, int niters) {
    // v.outc is num_outedges()
    // needs: v.num_inedges(), v.inedge(), v.id(), v.outc, v.set_data
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_of_subvertices) {
        return;
    }
    size_t i, outc = 0;
    double *inedge, *outedge;
    double sum = 0;

    inedge = in_graph + id;
    outedge = out_graph + id * num_of_vertices;
    id = st + id;

    for (i = 0; i < num_of_vertices; i++) {
        if (i != id && outedge[i] != 0) {
            outc++;
        }
    }

    if (iter == 0) {
        for (i = 0; i < num_of_vertices; i++) {
            if (i != id && outedge[i] != 0) {
                outedge[i] = 1.0 / outc;
                *updated = 1;
            }
        }
        vertices[id] = RANDOMRESETPROB;
    } else {
        for (i = 0; i < num_of_vertices; i++) {
            // we don't consider self-loop
            if (inedge[i * num_of_subvertices] && i != id) {
                sum += inedge[i * num_of_subvertices];
            }
        }
        double pagerank = (RANDOMRESETPROB + (1 - RANDOMRESETPROB) * sum);
        if (outc > 0) {
            double pagerankcont = pagerank / (double) outc;
            for (i = 0; i < num_of_vertices; i++) {
                if (i != id && outedge[i] != 0) {
                    outedge[i] = pagerankcont;
                    *updated = 1;
                }
            }
        } 
        vertices[id] = pagerank;
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
    // uint64_t i, st;
    struct fifo_entry *entry = NULL;

    while (1) {
        entry = (struct fifo_entry *) fifo_pop(conf->request_queue);
        if (entry) {
            sock_write_request(conf->res->req_sock, entry->id, entry->st, entry->st+1, entry->sub_m, entry->op, entry->which);
            sock_read_data(conf->res->req_sock);
            fifo_push(conf->fetching_queue, entry);
        } else {
            sock_write_request(conf->res->req_sock, -1, 0, 1, SUB_M, 0, 0);
            sock_read_data(conf->res->req_sock);
            fifo_close(conf->fetching_queue);
            break;
        }
    }
    // for (i = 0; i < NITERS; i++) {
    //     for (st = 0; st < M / SUB_M; st++) {
    //         sock_write_request(conf->res->req_sock, conf->id, st, st+1, SUB_M, 2, 0);
    //         sock_read_data(conf->res->req_sock);

    //         sock_write_request(conf->res->req_sock, conf->id, st, st+1, SUB_M, 3, 1);
    //         sock_read_data(conf->res->req_sock);
    //     }
    // }
    // sock_write_request(conf->res->req_sock, -1, st, st+1, SUB_M, 0, 0);
    // sock_read_data(conf->res->req_sock);
    return NULL;
}

int nds_pagerank(struct resources *res, int fd_in, int fd_out, uint64_t m, uint64_t sub_m) {
    size_t i, st;    
    double *outedges, *inedges;
    int temp;
    int updated_h, *updated_d;

    // result
    double *vertices, *vertices_d;

    size_t total_iteration;
    uint64_t stripe_size;

    struct timing_info *row_fetch_timing;
    struct timing_info *col_fetch_timing;
    struct timing_info *copy_in_timing;
    struct timing_info *pagerank_timing;
    struct timing_info *copy_update_timing;  
    struct timing_info *copy_out_timing;  
    struct timing_info *row_write_timing;  
    
    struct fetch_conf f_conf;

    struct timeval h_start, h_end;
    long duration;

    // initialization
    total_iteration = NITERS * (m / sub_m);

    row_fetch_timing = timing_info_new(total_iteration);
    if (row_fetch_timing == NULL) {
        printf("cannot create row_fetch_timing\n");
        return -1;
    }

    col_fetch_timing = timing_info_new(total_iteration);
    if (col_fetch_timing == NULL) {
        printf("cannot create col_fetch_timing\n");
        return -1;
    }

    row_write_timing = timing_info_new(total_iteration);
    if (row_write_timing == NULL) {
        printf("cannot create row_write_timing\n");
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

    copy_update_timing = timing_info_new(total_iteration);
    if (copy_update_timing == NULL) {
        printf("cannot create copy_update_timing\n");
        return -1;
    }

    copy_out_timing = timing_info_new(total_iteration);
    if (copy_out_timing == NULL) {
        printf("cannot create copy_out_timing\n");
        return -1;
    }

    // subgraph initialization
    stripe_size = m * sub_m * sizeof(double);
    cudaMalloc((void **) &outedges, stripe_size);
    cudaMalloc((void **) &inedges, stripe_size);
    cudaMalloc((void **) &updated_d, sizeof(int));

    // PR initialization
    vertices = (double *) calloc(sizeof(double), m);

    cudaMalloc((void **) &vertices_d, sizeof(double) * m);
    cudaMemset(vertices_d, 0, sizeof(double) * m);

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
    timing_info_set_starting_time(col_fetch_timing);
    timing_info_set_starting_time(row_write_timing);
    timing_info_set_starting_time(copy_in_timing);
    timing_info_set_starting_time(pagerank_timing);
    timing_info_set_starting_time(copy_out_timing);
    timing_info_set_starting_time(copy_update_timing);

    struct response *resp = NULL;

    gettimeofday(&h_start, NULL);
    // blockGEMM
    for (i = 0; i < NITERS; i++) {
        printf("iter: %lu\n", i);
        for (st = 0; st < m / sub_m; st++) {
            printf("st: %lu\n", st * sub_m);
            // outedges
            cudaMemcpyFromMmap(&f_conf, fd_out, st, st+1, sub_m, 2, 0,
                (char *) outedges, (char *) res->buf, stripe_size, row_fetch_timing);

            // inedges
            cudaMemcpyFromMmap(&f_conf, fd_in, st, st+1, sub_m, 3, 1,
                (char *) inedges, (char *) res->buf, stripe_size, col_fetch_timing);

            timing_info_push_start(pagerank_timing);
            updated_h = 0;
            cudaMemset(updated_d, 0, sizeof(int));
            pagerank_update<<<(sub_m+NUM_THREADS-1)/NUM_THREADS, NUM_THREADS>>>(vertices_d, st * sub_m, inedges, outedges, updated_d, m, sub_m, i, NITERS);
            timing_info_push_end(pagerank_timing);
            
            timing_info_push_start(copy_update_timing);
            cudaMemcpy(&updated_h, updated_d, sizeof(int), cudaMemcpyDeviceToHost);
            timing_info_push_end(copy_update_timing);
            if (updated_h) {
                // printf("updating st: %lu\n", st * sub_m);
                timing_info_push_start(copy_out_timing);
                cudaMemcpy(res->buf, outedges, stripe_size, cudaMemcpyDeviceToHost);
                timing_info_push_end(copy_out_timing);

                timing_info_push_start(row_write_timing);
                sock_write_request(res->req_sock, fd_out, st, st+1, sub_m, 4, 0);
                sock_read_data(res->req_sock);
    
                resp = sock_read_offset(res->sock);
                if (resp == NULL) {
                    fprintf(stderr, "sync error before RDMA ops\n");
                }
                free(resp);
                if (sock_write_data(res->sock)) { /* just send a dummy char back and forth */
                    fprintf(stderr, "sync error before RDMA ops\n");
                }
                timing_info_push_end(row_write_timing);
                // return_size = tensorstore_write_row_stripe_submatrix(&client, fd_out, st, st+1, num_of_subvertices);
            }
        }
        temp = fd_in;
        fd_in = fd_out;
        fd_out = temp;
    }
    sock_write_request(res->req_sock, -1, 0, 1, SUB_M, 0, 0);
    sock_read_data(res->req_sock);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);

    cudaMemcpy(vertices, vertices_d, sizeof(double) * m, cudaMemcpyDeviceToHost);
    
    printf("Pagerank duration: %f ms\n", (float) duration / 1000);    
    printf("Row fetch time: %f ms\n", (float) timing_info_duration(row_fetch_timing) / 1000);
    printf("Col fetch time: %f ms\n", (float) timing_info_duration(col_fetch_timing) / 1000);
    printf("Copy in time: %f ms\n", (float) timing_info_duration(copy_in_timing) / 1000);
    printf("Kernel time: %f ms\n", (float) timing_info_duration(pagerank_timing) / 1000);
    printf("copy update time: %f ms\n", (float) timing_info_duration(copy_update_timing) / 1000);
    printf("copy out time: %f ms\n", (float) timing_info_duration(copy_out_timing) / 1000);
    printf("Row write time: %f ms\n", (float) timing_info_duration(row_write_timing) / 1000);

    struct timestamps *tss = NULL;
    FILE *fptr;
    tss = timing_info_get_timestamps(row_fetch_timing);
    fptr = fopen("row_fetch_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(row_fetch_timing);

    tss = timing_info_get_timestamps(col_fetch_timing);
    fptr = fopen("col_fetch_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(col_fetch_timing);

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

    tss = timing_info_get_timestamps(copy_update_timing);
    fptr = fopen("copy_update_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(copy_update_timing);

    tss = timing_info_get_timestamps(copy_out_timing);
    fptr = fopen("copy_out_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(copy_out_timing);

    tss = timing_info_get_timestamps(row_write_timing);
    fptr = fopen("row_write_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(row_write_timing);

    fptr = fopen("log.txt", "w");
    for (i = 0; i < m; i++) {
        fprintf(fptr, "%lu %f\n", i, vertices[i]);
    }
    fclose(fptr);

    cudaFree(outedges);
    cudaFree(inedges);
    cudaFree(vertices_d);
    cudaFree(updated_d);
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
    uint64_t n, sub_n;
    int fd_in, fd_out;

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
    if (argc < 6) {
        printf("usage: %s <in-graph id> <out-graph id> <# of vertices> <# of subvertices> <port>\n", argv[0]);
        exit(1);
    } 

    fd_in = atoi(argv[1]);
    fd_out = atoi(argv[2]);
    n = (uint64_t) atoll(argv[3]);
    sub_n = (uint64_t) atoll(argv[4]);
    config.tcp_port = atoi(argv[5]);

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
    rc = nds_pagerank(&res, fd_in, fd_out, n, sub_n);

    close(res.sock);
    close(res.req_sock);
    munmap(hugepage_addr, BUF_SIZE);
    close(hugepage_fd);
    return rc;
}
