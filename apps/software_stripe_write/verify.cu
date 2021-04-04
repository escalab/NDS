extern "C" {
    #include "rdma.h"
    #include "timing.h"
    #include "fifo.h"
}

#define RANDOMRESETPROB 0.15
#define NUM_THREADS 1024

#define HUGEPAGE_SZ (4UL * 1024UL * 1024UL * 1024UL)
#define M 65536UL
#define SUB_M 256UL
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
    struct fifo *complete_queue;
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

int cudaMemcpyFromMmap(struct fetch_conf *conf, char *dst, const char *src, const size_t length, struct timing_info *fetch_timing) {
    struct response *res = NULL;

    timing_info_push_start(fetch_timing);
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

void *fetch_thread(void *args) {
    struct fetch_conf *conf = (struct fetch_conf *) args;
    // uint64_t i, st;
    uint64_t dsize = conf->m * conf->sub_m;
    double *ptr_a, *ptr_b;
    struct fifo_entry *entry = NULL;
    struct response *res = NULL;

    while (1) {
        entry = (struct fifo_entry *) fifo_pop(conf->fetching_queue);
        if (entry) {
            if (entry->op >= 4) {
                res = sock_read_offset(conf->res->sock);
                if (res == NULL) {
                    fprintf(stderr, "sync error before RDMA ops\n");
                }
                free(res);
                if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
                    fprintf(stderr, "sync error before RDMA ops\n");
                }
            } else {
                if (entry->which == 0) {
                    // ptr_a = conf->outedges + dsize * (count % IO_QUEUE_SZ);
                    ptr_a = conf->outedges;
                    cudaMemcpyFromMmap(conf, (char *) ptr_a, (char *) conf->hugepage_addr, dsize * sizeof(double), conf->row_fetch_timing);
                    entry->edges = ptr_a;
                } else {
                    // ptr_b = conf->inedges + dsize * (count % IO_QUEUE_SZ);
                    ptr_b = conf->inedges;
                    cudaMemcpyFromMmap(conf, (char *) ptr_b, (char *) conf->hugepage_addr, dsize * sizeof(double), conf->col_fetch_timing);
                    entry->edges = ptr_b;
                }
            }
            fifo_push(conf->kernel_queue, entry);
        } else {
            fifo_close(conf->kernel_queue);
            break;
        }
    }

    // for (i = 0; i < NITERS; i++) {
    //     for (st = 0; st < conf->m / conf->sub_m; st++) {
    //         // while (fifo_empty(conf->complete_queue)) {

    //         // }
    //         entry = (struct fifo_entry *) fifo_pop(conf->complete_queue);
    //         ptr_a = conf->outedges + dsize * (count % IO_QUEUE_SZ);
    //         ptr_b = conf->inedges + dsize * (count % IO_QUEUE_SZ);

    //         cudaMemcpyFromMmap(conf, (char *) ptr_a, (char *) conf->hugepage_addr, dsize * sizeof(int64_t), conf->row_fetch_timing);
    //         cudaMemcpyFromMmap(conf, (char *) ptr_b, (char *) conf->hugepage_addr, dsize * sizeof(int64_t), conf->col_fetch_timing);
    //         count++;

    //         entry->outedges = ptr_a;
    //         entry->inedges = ptr_b;
    //         fifo_push(conf->fetching_queue, entry);
    //     }
    // }
    return NULL;
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

int nds_stripe_write(struct resources *res, int matrix_id, uint64_t m, uint64_t sub_m) {
    size_t i, st;    
    double *outedges;

    size_t total_iteration;
    uint64_t stripe_size;

    struct fifo *request_queue;
    struct fifo *fetching_queue;
    struct fifo *kernel_queue;
    struct fifo *complete_queue; 
    struct fifo_entry *entries = (struct fifo_entry *) calloc(IO_QUEUE_SZ, sizeof(struct fifo_entry));
    struct fifo_entry *entry = NULL;

    struct timing_info *copy_out_timing;  
    struct timing_info *row_write_timing;  
    
    pthread_t f_thread_id; 
    struct fetch_conf f_conf;

    pthread_t r_thread_id; 
    struct request_conf r_conf;

    struct timeval h_start, h_end;
    long duration;

    // initialization
    total_iteration = NITERS * (m / sub_m);

    copy_out_timing = timing_info_new(total_iteration);
    if (copy_out_timing == NULL) {
        printf("cannot create copy_out_timing\n");
        return -1;
    }

    row_write_timing = timing_info_new(total_iteration);
    if (row_write_timing == NULL) {
        printf("cannot create row_write_timing\n");
        return -1;
    }

    // it causes problem if size == 1
    request_queue = fifo_new(IO_QUEUE_SZ * 2);
	if (request_queue == NULL) {
        printf("cannot create request_queue\n");
        return -1;
    }

    fetching_queue = fifo_new(IO_QUEUE_SZ * 2);
	if (fetching_queue == NULL) {
        printf("cannot create fetching_queue\n");
        return -1;
    }

    kernel_queue = fifo_new(IO_QUEUE_SZ * 2);
	if (fetching_queue == NULL) {
        printf("cannot create fetching_queue\n");
        return -1;
    }
    
    complete_queue = fifo_new(IO_QUEUE_SZ * 2);
	if (complete_queue == NULL) {
        printf("cannot create complete_queue\n");
        return -1;
    }
    
    for (i = 0; i < IO_QUEUE_SZ; i++) {
        fifo_push(complete_queue, entries + i);
    }

    // subgraph initialization
    stripe_size = m * sub_m * sizeof(double);
    cudaMalloc((void **) &outedges, stripe_size);
    cudaMemset(outedges, 5, stripe_size);

    r_conf.res = res;
    r_conf.request_queue = request_queue;
    r_conf.fetching_queue = fetching_queue;

    // create thread here
    f_conf.res = res;
    f_conf.m = m;
    f_conf.sub_m = sub_m;
    f_conf.outedges = outedges;
    f_conf.hugepage_addr = res->buf;
    f_conf.fetching_queue = fetching_queue;
    f_conf.kernel_queue = kernel_queue;
    f_conf.complete_queue = complete_queue;

    timing_info_set_starting_time(copy_out_timing);
    timing_info_set_starting_time(row_write_timing);

    gettimeofday(&h_start, NULL);
    pthread_create(&r_thread_id, NULL, request_thread, &r_conf); 
	pthread_create(&f_thread_id, NULL, fetch_thread, &f_conf); 
    // blockGEMM
    for (i = 0; i < NITERS; i++) {
        // printf("iter: %lu\n", i);
        for (st = 0; st < m / sub_m; st++) {
            // printf("st: %lu\n", st * sub_m);
            // printf("updating st: %lu\n", st * sub_m);
            timing_info_push_start(copy_out_timing);
            // assume we can write to different offset on RDMA buffer 
            // (it looks I/O bound anyway)
            // cudaMemcpy(res->buf, outedges, stripe_size, cudaMemcpyDeviceToHost);
            timing_info_push_end(copy_out_timing);

            timing_info_push_start(row_write_timing);
            entry = (struct fifo_entry *) fifo_pop(complete_queue);
            entry->id = matrix_id;
            entry->op = 5;
            entry->st = st;
            entry->sub_m = sub_m;
            entry->which = 0;
            fifo_push(request_queue, entry);
            entry = (struct fifo_entry *) fifo_pop(kernel_queue);
            fifo_push(complete_queue, entry);
            timing_info_push_end(row_write_timing);
        }
    }
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);

    fifo_close(request_queue);
    pthread_join(r_thread_id, NULL); 
    pthread_join(f_thread_id, NULL); 
    
    printf("Pagerank duration: %f ms\n", (float) duration / 1000);    
    printf("copy out time: %f ms\n", (float) timing_info_duration(copy_out_timing) / 1000);
    printf("Row write time: %f ms\n", (float) timing_info_duration(row_write_timing) / 1000);

    timing_info_free(copy_out_timing);
    timing_info_free(row_write_timing);

    cudaFree(outedges);

    fifo_free(kernel_queue);
    fifo_free(request_queue);
    fifo_free(fetching_queue);
    fifo_free(complete_queue);
    free(entries);
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
    int matrix_id;

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
        printf("usage: %s <id> <# of vertices> <# of subvertices> <port>\n", argv[0]);
        exit(1);
    } 

    matrix_id = atoi(argv[1]);
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
    rc = nds_stripe_write(&res, matrix_id, n, sub_n);

    close(res.sock);
    close(res.req_sock);
    munmap(hugepage_addr, BUF_SIZE);
    close(hugepage_fd);
    return rc;
}
