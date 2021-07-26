#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

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
#define M 32768UL
#define SUB_M 16384UL
#define AGGREGATED_SZ (SUB_M * SUB_M * 8UL)

#define IO_QUEUE_SZ (HUGEPAGE_SZ / AGGREGATED_SZ)
// #define IO_QUEUE_SZ 2UL

void print_config(struct config_t config);

struct fetch_conf {
    struct resources *res;
    uint64_t m, sub_m;
    char *hugepage_addr;
    struct fifo *request_queue;
    struct fifo *io_queue;
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

    for (i = 0; i < conf->m / conf->sub_m; i++) {
        for (j = 0; j < conf->m / conf->sub_m; j++) {
            entry = (struct fifo_entry *) fifo_pop(conf->request_queue);
            entry->resp = sock_read_offset(conf->res->sock);
            if (entry->resp == NULL) {
                fprintf(stderr, "sync error before RDMA ops\n");
                return NULL;
            }
        
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
    uint64_t iter, i, j;
    struct fifo_entry *entry = NULL;
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

uint64_t aggregated_verify(double *arr, double *answer, uint64_t x, uint64_t y, uint64_t m, uint64_t sub_m) {
    uint64_t i, j, error = 0;
    int rc = 0;
    // double *arr_ptr = arr;
    double *answer_ptr = answer + x * sub_m + y * m * sub_m;
    for (i = 0; i < sub_m; i++) {
        // rc = rc | memcmp(arr_ptr, answer_ptr, sizeof(double) * sub_m);
        // arr_ptr += sub_m;
        // answer_ptr += m;
        for (j = 0; j < sub_m; j++) {
            if (arr[i * sub_m + j] != answer_ptr[i * m + j]) {
                error++;
            }
        }
    }
    return error;
}

int nds_aggregated_verify(struct resources *res, uint64_t id, uint64_t m, uint64_t sub_m, double *answer) {
    size_t i, j;    
    double *aggregated_matrix;
    
    uint64_t dsize, error = 0;
    int rc = 0;

    struct fifo *request_queue;
    struct fifo *io_queue;
    struct fifo *complete_queue; 
    struct fifo_entry *entries = (struct fifo_entry *) calloc(IO_QUEUE_SZ, sizeof(struct fifo_entry));
    struct fifo_entry *entry = NULL;

    pthread_t f_thread_id; 
    struct fetch_conf f_conf;

    pthread_t r_thread_id; 
    struct request_conf r_conf;

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

    // subgraph initialization
    dsize = SUB_M * SUB_M * sizeof(double);
    aggregated_matrix = (double *) malloc(dsize);
    memset(aggregated_matrix, 0, dsize);

    r_conf.res = res;
    r_conf.id = id;
    r_conf.m = M;
    r_conf.sub_m = SUB_M;
    r_conf.request_queue = request_queue;
    r_conf.complete_queue = complete_queue;

    // create thread here
    f_conf.res = res;
    f_conf.m = M;
    f_conf.sub_m = SUB_M;
    f_conf.io_queue = io_queue;
    f_conf.request_queue = request_queue;

    pthread_create(&r_thread_id, NULL, request_thread, &r_conf); 
    pthread_create(&f_thread_id, NULL, fetch_thread, &f_conf);  

    // blockGEMM
    for (i = 0; i < (M / SUB_M); i++) {
        for (j = 0; j < (M / SUB_M); j++) {
            entry = (struct fifo_entry *) fifo_pop(io_queue);

            // timing_info_push_start(copy_in_timing);
            // cudaMemcpy(d_addr, res->buf + entry->resp->offset, SUB_M * SUB_M * sizeof(double), cudaMemcpyHostToDevice);
            // timing_info_push_end(copy_in_timing);
            memcpy(aggregated_matrix, res->buf + entry->resp->offset, dsize);

            // rc = rc | aggregated_verify(aggregated_matrix, answer, j, i, m, sub_m);
            error += aggregated_verify(aggregated_matrix, answer, j, i, m, sub_m);
            free(entry->resp);
            fifo_push(complete_queue, entry);
        }
    }

    printf("result: %lu\n", error);

    pthread_join(r_thread_id, NULL); 
    pthread_join(f_thread_id, NULL); 

    free(aggregated_matrix);
    fifo_free(request_queue);
    fifo_free(io_queue);
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
    uint64_t matrix_id;
    int hugepage_fd;
    char *hugepage_addr;

    int answer_fd;
    char *answer_addr;

    // RDMA
    struct resources res;
    struct config_t config = {
        "mlx4_0",  /* dev_name */
        "127.0.0.1",  /* server_name */
        19875, /* tcp_port */
        1,     /* ib_port */
        0     /* gid_idx */
    };

    if (argc < 4) {
        printf("usage: %s <matrix_id> <answer path> <port>\n", argv[0]);
        exit(1);
    } 
    matrix_id = (uint64_t) atoll(argv[1]);
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

    answer_fd = open(argv[2], O_RDONLY, 0644);
    if (answer_fd < 0) {
        perror("open");
        exit(1);
    }

    answer_addr = (char *) mmap(0, M*M*sizeof(double), PROT_READ, MAP_PRIVATE, answer_fd, 0);
    if (answer_addr==MAP_FAILED) {
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
    rc = nds_aggregated_verify(&res, matrix_id, M, SUB_M, (double *) answer_addr);
    
    close(res.sock);
    close(res.req_sock);
    munmap(hugepage_addr, BUF_SIZE);
    close(hugepage_fd);
    munmap(answer_addr, M*M*sizeof(double));
    close(answer_fd);
    return rc;
}
