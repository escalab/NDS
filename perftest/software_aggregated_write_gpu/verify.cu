extern "C" {
    #include "rdma.h"
    #include "timing.h"
    #include "fifo.h"
}

#define HUGEPAGE_SZ (4UL * 1024UL * 1024UL * 1024UL)
#define M 65536UL
#define SUB_M 16384UL
#define AGGREGATED_SZ (SUB_M * SUB_M * 8UL)

#define IO_QUEUE_SZ (HUGEPAGE_SZ / AGGREGATED_SZ)
// #define IO_QUEUE_SZ 16UL

#define AGGREGATED_WRITE 2
#define CURRENT_OP AGGREGATED_WRITE
#define NITERS 4UL

void print_config(struct config_t config);

struct fetch_conf {
    struct resources *res;
    uint64_t m, sub_m;
    double *d_addr, *inedges;
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
    size_t i, j;
    double *edges;
};

void *fetch_thread(void *args) {
    struct fetch_conf *conf = (struct fetch_conf *) args;
    uint64_t iter, i, j;
    struct fifo_entry *entry = NULL;
    struct response *res = NULL;

    for (iter = 0; iter < NITERS; iter++) {
        for (i = 0; i < (M / SUB_M); i++) {
            for (j = 0; j < (M / SUB_M); j++) {
                entry = (struct fifo_entry *) fifo_pop(conf->fetching_queue);
                if (entry) {
                    if (entry->op >= 2) {
                        res = sock_read_offset(conf->res->sock);
                        if (res == NULL) {
                            fprintf(stderr, "sync error before RDMA ops\n");
                        }
                        free(res);
                        if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
                            fprintf(stderr, "sync error before RDMA ops\n");
                        }
                        // printf("done with entry->i: %lu, entry->j: %lu\n", entry->i, entry->j);
                        fifo_push(conf->complete_queue, entry);
                    } 
                }        
            }
        }
    }
    printf("fetch thread close\n");
    return NULL;
}

void *request_thread(void *args) {
    struct request_conf *conf = (struct request_conf *) args;
    uint64_t iter, i, j;
    struct fifo_entry *entry = NULL;

    for (iter = 0; iter < NITERS; iter++) {
        for (i = 0; i < (M / SUB_M); i++) {
            for (j = 0; j < (M / SUB_M); j++) {
                entry = (struct fifo_entry *) fifo_pop(conf->request_queue);
                if (entry) {
                    sock_write_request(conf->res->req_sock, entry->id, entry->j, entry->i, entry->sub_m, entry->op, entry->which);
                    sock_read_data(conf->res->req_sock);
                    fifo_push(conf->fetching_queue, entry);
                }
        
            }
        }
    }
    printf("request thread close\n");
    return NULL;
}

int nds_aggregated_write(struct resources *res, int matrix_id, uint64_t m, uint64_t sub_m) {
    size_t iter, i, j;    
    double *d_addr, *h_addr;

    size_t total_iteration;

    struct fifo *request_queue;
    struct fifo *fetching_queue;
    struct fifo *kernel_queue;
    struct fifo *complete_queue; 
    struct fifo_entry *entries = (struct fifo_entry *) calloc(IO_QUEUE_SZ, sizeof(struct fifo_entry));
    struct fifo_entry *entry = NULL;

    struct timing_info *copy_out_timing;  
    struct timing_info *write_timing;  
    
    pthread_t f_thread_id; 
    struct fetch_conf f_conf;

    pthread_t r_thread_id; 
    struct request_conf r_conf;

    struct timeval h_start, h_end;
    long duration;

    // initialization
    total_iteration = NITERS * (m / sub_m) * (m / sub_m);

    copy_out_timing = timing_info_new(total_iteration);
    if (copy_out_timing == NULL) {
        printf("cannot create copy_out_timing\n");
        return -1;
    }

    write_timing = timing_info_new(total_iteration);
    if (write_timing == NULL) {
        printf("cannot create write_timing\n");
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
    h_addr = (double *) malloc(HUGEPAGE_SZ);
    cudaMalloc((void **) &d_addr, HUGEPAGE_SZ);
    for (i = 0; i < HUGEPAGE_SZ / sizeof(double); i++) {
        h_addr[i] = (double) rand() / (double) RAND_MAX;
    }
    cudaMemcpy(d_addr, h_addr, HUGEPAGE_SZ, cudaMemcpyHostToDevice);

    r_conf.res = res;
    r_conf.request_queue = request_queue;
    r_conf.fetching_queue = fetching_queue;

    // create thread here
    f_conf.res = res;
    f_conf.m = m;
    f_conf.sub_m = sub_m;
    f_conf.d_addr = d_addr;
    f_conf.hugepage_addr = res->buf;
    f_conf.fetching_queue = fetching_queue;
    f_conf.kernel_queue = kernel_queue;
    f_conf.complete_queue = complete_queue;

    timing_info_set_starting_time(copy_out_timing);
    timing_info_set_starting_time(write_timing);

    gettimeofday(&h_start, NULL);
    pthread_create(&r_thread_id, NULL, request_thread, &r_conf); 
	pthread_create(&f_thread_id, NULL, fetch_thread, &f_conf); 
    // blockGEMM
    uint64_t count = 0;
    for (iter = 0; iter < NITERS; iter++) {
        for (i = 0; i < (M / SUB_M); i++) {
            for (j = 0; j < (M / SUB_M); j++) {
                // printf("i: %lu, j: %lu\n", i, j);

                // assume we can write to different offset on RDMA buffer 
                // (it looks I/O bound anyway)
                entry = (struct fifo_entry *) fifo_pop(complete_queue);
                timing_info_push_start(copy_out_timing);
                cudaMemcpy(res->buf + (AGGREGATED_SZ * (count % IO_QUEUE_SZ)), (char *) d_addr + (AGGREGATED_SZ * (count % IO_QUEUE_SZ)), AGGREGATED_SZ, cudaMemcpyDeviceToHost);
                timing_info_push_end(copy_out_timing);
                count++;
                entry->id = matrix_id;
                entry->op = CURRENT_OP;
                entry->i = i;
                entry->j = j;
                entry->sub_m = SUB_M;
                entry->which = 0;
                fifo_push(request_queue, entry);
            }
        }
    }
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);

    // fifo_close(request_queue);
    pthread_join(r_thread_id, NULL); 
    sock_write_request(res->req_sock, -1, 0, 1, SUB_M, CURRENT_OP, 0);
    sock_read_data(res->req_sock);
    pthread_join(f_thread_id, NULL); 
    
    printf("Pagerank duration: %f ms\n", (float) duration / 1000);    
    printf("copy out time: %f ms\n", (float) timing_info_duration(copy_out_timing) / 1000);
    printf("Row write time: %f ms\n", (float) timing_info_duration(write_timing) / 1000);

    timing_info_free(copy_out_timing);
    timing_info_free(write_timing);

    cudaFree(d_addr);

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
    if (argc < 3) {
        printf("usage: %s <id> <port>\n", argv[0]);
        exit(1);
    } 

    matrix_id = atoi(argv[1]);
    config.tcp_port = atoi(argv[2]);

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
    rc = nds_aggregated_write(&res, matrix_id, M, SUB_M);

    close(res.sock);
    close(res.req_sock);
    munmap(hugepage_addr, BUF_SIZE);
    close(hugepage_fd);

    return rc;
}
