#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include "rdma.h"
#include "timing.h"
#include "fifo.h"


#define HUGEPAGE_SZ (4UL * 1024UL * 1024UL * 1024UL)
#define M 65536UL
#define SUB_M 4096UL
#define AGGREGATED_SZ (SUB_M * SUB_M * 8UL)

#define IO_QUEUE_SZ (HUGEPAGE_SZ / AGGREGATED_SZ)
// #define IO_QUEUE_SZ 2UL

void print_config(struct config_t config);

struct fetch_conf {
    struct resources *res;
    uint64_t m, sub_m;
    struct fifo *request_queue;
    struct fifo *io_queue;
    struct timing_info *aggregated_fetch_timing;
    char *out_addr;
};

struct request_conf {
    struct resources *res;
    struct fifo *complete_queue;
    struct fifo *request_queue;
    uint64_t id;
    uint64_t m, sub_m;
};

struct fifo_entry {
    struct response *resp;
};

void *fetch_thread(void *args) {
    struct fetch_conf *conf = (struct fetch_conf *) args;
    struct fifo_entry *entry = NULL;
    uint64_t iter, i, j;
    char *d_addr = conf->out_addr;

    for (i = 0; i < conf->m / conf->sub_m; i++) {
        for (j = 0; j < conf->m / conf->sub_m; j++) {
            entry = (struct fifo_entry *) fifo_pop(conf->request_queue);
            entry->resp = sock_read_offset(conf->res->sock);
            if (entry->resp == NULL) {
                fprintf(stderr, "sync error before RDMA ops\n");
                return NULL;
            }
            memcpy(d_addr, conf->res->buf + entry->resp->offset, SUB_M * SUB_M * sizeof(double));
            // msync(conf->out_addr, M * M * sizeof(double), MS_SYNC);
            d_addr += SUB_M * SUB_M * sizeof(double);
            if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
                fprintf(stderr, "sync error before RDMA ops\n");
                return NULL;
            }
            fifo_push(conf->io_queue, entry);
        }
    }
    return NULL;
}

void *request_thread(void *args) {
    struct request_conf *conf = (struct request_conf *) args;
    struct fifo_entry *entry = NULL;
    uint64_t i, j;
    for (i = 0; i < conf->m / conf->sub_m; i++) {
        for (j = 0; j < conf->m / conf->sub_m; j++) {
            entry = (struct fifo_entry *) fifo_pop(conf->complete_queue);
            sock_write_request(conf->res->req_sock, conf->id, j, i, SUB_M, 1, 0);
            sock_read_data(conf->res->req_sock);
            fifo_push(conf->request_queue, entry);
        }
    }    
    // send a signal to tell storage backend the iteration is done.
    sock_write_request(conf->res->req_sock, -1, 0, 0, SUB_M, 1, 0);
    sock_read_data(conf->res->req_sock);
    return NULL;
}

int nds_aggregated_oracle_datagen(struct resources *res, uint64_t id, char *out_addr) {
    size_t iter, i, j;

    struct fifo *request_queue;
    struct fifo *io_queue;
    struct fifo *complete_queue; 
    struct fifo_entry *entries = (struct fifo_entry *) calloc(IO_QUEUE_SZ, sizeof(struct fifo_entry));
    struct fifo_entry *entry = NULL;  
    
    pthread_t f_thread_id; 
    struct fetch_conf f_conf;

    pthread_t r_thread_id; 
    struct request_conf r_conf;

    struct timeval h_start, h_end;
    long duration;
    
    struct timing_info *aggregated_fetch_timing;
    struct timing_info *copy_in_timing;

    size_t total_iteration = (M / SUB_M) * (M / SUB_M);
    aggregated_fetch_timing = timing_info_new(total_iteration);
    if (aggregated_fetch_timing == NULL) {
        printf("cannot create aggregated_fetch_timing\n");
        return -1;
    }

    copy_in_timing = timing_info_new(total_iteration);
    if (copy_in_timing == NULL) {
        printf("cannot create copy_in_timing\n");
        return -1;
    }

    // it causes problem if size == 1
    request_queue = fifo_new(IO_QUEUE_SZ * 2);
	if (request_queue == NULL) {
        printf("cannot create request_queue\n");
        return -1;
    }

    io_queue = fifo_new(IO_QUEUE_SZ * 2);
	if (io_queue == NULL) {
        printf("cannot create io_queue\n");
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

    // create thread here
    r_conf.res = res;
    r_conf.id = id;
    r_conf.m = M;
    r_conf.sub_m = SUB_M;
    r_conf.request_queue = request_queue;
    r_conf.complete_queue = complete_queue;

    f_conf.res = res;
    f_conf.m = M;
    f_conf.sub_m = SUB_M;
    f_conf.io_queue = io_queue;
    f_conf.request_queue = request_queue;
    f_conf.aggregated_fetch_timing = aggregated_fetch_timing;
    f_conf.out_addr = out_addr;

    gettimeofday(&h_start, NULL);
    pthread_create(&r_thread_id, NULL, request_thread, &r_conf); 
    pthread_create(&f_thread_id, NULL, fetch_thread, &f_conf); 

    // blockGEMM
    for (i = 0; i < (M / SUB_M); i++) {
        for (j = 0; j < (M / SUB_M); j++) {
            entry = (struct fifo_entry *) fifo_pop(io_queue);

            // timing_info_push_start(copy_in_timing);
            // cudaMemcpy(d_addr, res->buf + entry->resp->offset, SUB_M * SUB_M * sizeof(double), cudaMemcpyHostToDevice);
            // timing_info_push_end(copy_in_timing);
            free(entry->resp);
            fifo_push(complete_queue, entry);
        }
    }

    pthread_join(r_thread_id, NULL); 
    pthread_join(f_thread_id, NULL); 

    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    
    printf("Aggregated Read End-to-end duration: %f ms\n", (float) duration / 1000);    
    printf("Col fetch time: %f ms\n", (float) timing_info_duration(aggregated_fetch_timing) / 1000);
    printf("Copy in time: %f ms\n", (float) timing_info_duration(copy_in_timing) / 1000);
    timing_info_free(aggregated_fetch_timing);
    timing_info_free(copy_in_timing);

    // Memory clean-up and CUBLAS shutdown
    free(entries);
    fifo_free(request_queue);
    fifo_free(io_queue);
    fifo_free(complete_queue);    
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
    uint64_t matrix_id;
    int hugepage_fd;
    char *hugepage_addr;

    int ret;
    int out_fd;
    char *out_addr;
    size_t dsize = sizeof(double) * M * M;

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
    if (argc < 4) {
        printf("usage: %s <outfile path> <matrix_id> <port>\n", argv[0]);
        exit(1);
    } 
    matrix_id = (uint64_t) atoll(argv[2]);
    config.tcp_port = atoi(argv[3]);

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

    out_fd = open(argv[1], O_RDWR | O_CREAT | O_TRUNC, 0644);
    ret = posix_fallocate(out_fd, 0, dsize);
    if (ret != 0) {
        printf("fallocate error code: %d\n", ret);
        exit(ret);
    }
    out_addr = (char *) mmap(NULL, dsize, PROT_READ | PROT_WRITE, MAP_SHARED, out_fd, 0);

    printf("calculating the result of pagerank\n");
    rc = nds_aggregated_oracle_datagen(&res, matrix_id, out_addr);

    close(res.sock);
    close(res.req_sock);
    munmap(hugepage_addr, BUF_SIZE);
    close(hugepage_fd);
    return rc;
}
