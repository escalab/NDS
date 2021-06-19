#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <cuda.h>
#include "cublas_v2.h"

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

extern "C" {
    #include "rdma.h"
    #include "timing.h"
    #include "fifo.h"
}

#define HUGEPAGE_SZ (4UL * 1024UL * 1024UL * 1024UL)
#define M 65536UL
#define SUB_M 512UL
#define AGGREGATED_SZ (SUB_M * SUB_M * 8UL)

#define IO_QUEUE_SZ (HUGEPAGE_SZ / AGGREGATED_SZ)
// #define IO_QUEUE_SZ 1UL

#define NITERS 4UL

void print_config(struct config_t config);

struct fetch_conf {
    struct resources *res;
    uint64_t m, sub_m;
    char *hugepage_addr;
    struct fifo *sending_queue;
    struct fifo *complete_queue;
    struct timing_info *aggregated_fetch_timing;
};

struct request_conf {
    struct resources *res;
    uint64_t id;
    uint64_t m, sub_m;
};

struct fifo_entry {
    struct response *resp;
};

// int cudaMemcpyFromMmap(struct fetch_conf *conf, char *dst, const char *src, const size_t length, struct timing_info *fetch_timing) {
//     struct response *res = NULL;

//     timing_info_push_start(fetch_timing);
//     res = sock_read_offset(conf->res->sock);
//     if (res == NULL) {
//         fprintf(stderr, "sync error before RDMA ops\n");
//         return 1;
//     }

//     // if (res->id == 0) {
//     //     printf("fetching row [%lu:%lu]\n", res->x, res->y);
//     // } else {
//     //     printf("fetching col [%lu:%lu]\n", res->x, res->y);
//     // }
//     // printf("offset: %lu\n", res->offset);

//     timing_info_push_end(fetch_timing);

//     timing_info_push_start(conf->copy_in_timing);
//     cudaMemcpy(dst, src + res->offset, length, cudaMemcpyHostToDevice);
//     timing_info_push_end(conf->copy_in_timing);

//     free(res);
//     if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
//         fprintf(stderr, "sync error before RDMA ops\n");
//         return 1;
//     }
//     return 0;
// }

void *fetch_thread(void *args) {
    struct fetch_conf *conf = (struct fetch_conf *) args;
    struct fifo_entry *entry = NULL;
    uint64_t iter, i, j;

    for (iter = 0; iter < NITERS; iter++) {
        for (i = 0; i < conf->m / conf->sub_m; i++) {
            for (j = 0; j < conf->m / conf->sub_m; j++) {
                entry = (struct fifo_entry *) fifo_pop(conf->complete_queue);
                timing_info_push_start(conf->aggregated_fetch_timing);
                entry->resp = sock_read_offset(conf->res->sock);
                if (entry->resp == NULL) {
                    fprintf(stderr, "sync error before RDMA ops\n");
                    return NULL;
                }
            
                if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
                    fprintf(stderr, "sync error before RDMA ops\n");
                    return NULL;
                }
                timing_info_push_end(conf->aggregated_fetch_timing);
                fifo_push(conf->sending_queue, entry);
            }
        }
    }
    return NULL;
}

void *request_thread(void *args) {
    struct request_conf *conf = (struct request_conf *) args;
    uint64_t iter, i, j;
    for (iter = 0; iter < NITERS; iter++) {
        for (i = 0; i < conf->m / conf->sub_m; i++) {
            for (j = 0; j < conf->m / conf->sub_m; j++) {
                sock_write_request(conf->res->req_sock, conf->id, j, i, SUB_M, 1, 0);
                sock_read_data(conf->res->req_sock);
            }
        }
    }
    // send a signal to tell storage backend the iteration is done.
    sock_write_request(conf->res->req_sock, -1, 0, 0, SUB_M, 1, 0);
    sock_read_data(conf->res->req_sock);
    return NULL;
}

int nds_aggregated_read(struct resources *res, uint64_t id) {
    size_t iter, i, j;
    double *d_addr;

    struct fifo *sending_queue;
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

    size_t total_iteration = (M / SUB_M) * (M / SUB_M) * NITERS;
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

    cudaMalloc(&d_addr, SUB_M * SUB_M * sizeof(double));

    // it causes problem if size == 1
    sending_queue = fifo_new(IO_QUEUE_SZ * 2);
	if (sending_queue == NULL) {
        printf("cannot create sending_queue\n");
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

    r_conf.res = res;
    r_conf.id = id;
    r_conf.m = M;
    r_conf.sub_m = SUB_M;

    // create thread here
    f_conf.res = res;
    f_conf.m = M;
    f_conf.sub_m = SUB_M;
    f_conf.sending_queue = sending_queue;
    f_conf.complete_queue = complete_queue;
    f_conf.aggregated_fetch_timing = aggregated_fetch_timing;

    gettimeofday(&h_start, NULL);
    pthread_create(&r_thread_id, NULL, request_thread, &r_conf); 
    pthread_create(&f_thread_id, NULL, fetch_thread, &f_conf); 

    // blockGEMM
    for (iter = 0; iter < NITERS; iter++) {
        for (i = 0; i < (M / SUB_M); i++) {
            for (j = 0; j < (M / SUB_M); j++) {
                entry = (struct fifo_entry *) fifo_pop(sending_queue);

                // timing_info_push_start(copy_in_timing);
                // cudaMemcpy(d_addr, res->buf + entry->resp->offset, SUB_M * SUB_M * sizeof(double), cudaMemcpyHostToDevice);
                // timing_info_push_end(copy_in_timing);
                free(entry->resp);
                fifo_push(complete_queue, entry);
            }
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
    cudaFree(d_addr);
    fifo_free(complete_queue);
    fifo_free(sending_queue);
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
        printf("usage: %s <matrix_id> <port>\n", argv[0]);
        exit(1);
    } 
    matrix_id = (uint64_t) atoll(argv[1]);
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

    // use the first res to find the device.
    if (resources_find_device(&res, &config)) {
        fprintf(stderr, "failed to find a device for resources\n");
        exit(1);
    }

    if (resources_create(&res, &config)) {
        fprintf(stderr, "failed to create resources\n");
        exit(1);
    }

    /* connect the QPs */
    if (connect_qp(&res, &config)) {
        fprintf(stderr, "failed to connect QPs\n");
        exit(1);
    }

    printf("calculating the result of pagerank\n");
    rc = nds_aggregated_read(&res, matrix_id);

    munmap(hugepage_addr, BUF_SIZE);
    if (resources_destroy(&res)) {
        fprintf(stderr, "failed to destroy resources\n");
        exit(1);
    }
    close(res.req_sock);
    close(hugepage_fd);
    return rc;
}
