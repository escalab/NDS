extern "C" {
    #include "rdma.h"
    #include "fifo.h"
}

#include <fcntl.h>
#include <unistd.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define M 1536UL
#define N M
#define K M

#define LDB1 M
#define LDB2 LDB1*N
#define LDC1 N

#define SUB_M 512UL

#define HUGEPAGE_SZ (4UL * 1024UL * 1024UL * 1024UL)
#define AGGREGATED_SZ (SUB_M * SUB_M * SUB_M * 8UL)

// #define IO_QUEUE_SZ (HUGEPAGE_SZ / AGGREGATED_SZ / 2UL)
#define IO_QUEUE_SZ 1UL

#define NITERS 4UL

void print_config(struct config_t config);

struct fetch_conf {
    struct resources *res;
    uint64_t m, sub_m;
    double *sub_B;
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
    double *sub_B;
};

int memcpyFromMmap(struct fetch_conf *conf, char *dst, const char *src, const size_t length) {
    struct response *res = NULL;

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

    memcpy(dst, src + res->offset, length);

    free(res);
    if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
        fprintf(stderr, "sync error before RDMA ops\n");
        return 1;
    }
    return 0;
}

void *fetch_thread(void *args) {
    struct fetch_conf *conf = (struct fetch_conf *) args;
    uint64_t n, m, k;
    uint64_t dsize = SUB_M * SUB_M * SUB_M;
    double *ptr;
    struct fifo_entry *entry = NULL;
    uint64_t count = 0;

    for (n = 0; n < N / SUB_M; n++) {
        for (m = 0; m < M / SUB_M; m++) {
            for (k = 0; k < K / SUB_M; k++) {
                entry = (struct fifo_entry *) fifo_pop(conf->request_queue);
                ptr = conf->sub_B + dsize * (count % IO_QUEUE_SZ);

                memcpyFromMmap(conf, (char *) ptr, (char *) conf->res->buf, dsize * sizeof(double));
                count++;

                entry->sub_B = ptr;
                fifo_push(conf->io_queue, entry);
            }
        }
    }
    return NULL;
}

void *request_thread(void *args) {
    struct request_conf *conf = (struct request_conf *) args;
    uint64_t n, m, k;
    struct fifo_entry *entry = NULL;
    for (n = 0; n < N / SUB_M; n++) {
        for (m = 0; m < M / SUB_M; m++) {
            for (k = 0; k < K / SUB_M; k++) {
                entry = (struct fifo_entry *) fifo_pop(conf->complete_queue);
                // TODO: need to add one more param
                sock_write_request(conf->res->req_sock, conf->id, n, m, SUB_M, 4, k);
                sock_read_data(conf->res->req_sock);
                fifo_push(conf->request_queue, entry);
            }
        }
    }

    sock_write_request(conf->res->req_sock, -1, n, m, k, 1, 0);
    sock_read_data(conf->res->req_sock);
    return NULL;
}

void generate_data(double *array, uint64_t size) {
    uint64_t i;
    for (i = 0; i < size; i++) {
        // array[i] = (((double) rand())/RAND_MAX - 0.5)*100;
        array[i] = i;        
    }
}

uint64_t verify(const double *answer_C, const double *C, uint64_t size) {
    uint64_t i, error = 0;
    for (i = 0; i < size; i++) {
        if (answer_C[i] != C[i]) {
            printf("index %lu error, answer_C: %f, C: %f\n", i, answer_C[i], C[i]);
            error++;
            return error;
        }
    }
    return error;
}

void reassemble_block_tensor_from_seq(const double *seq_matrix, double *result_matrix, size_t n, size_t sub_n, size_t i, size_t j, size_t k) {
    size_t ii, jj, kk, a, b, c;
    for (ii = i, a = 0; ii < i+sub_n; ii++, a++) {
        for (jj = j, b = 0; jj < j+sub_n; jj++, b++) {
            // printf("i=%lu, j=%lu, k=%lu, result_offset=%lu, seq_offset=%lu\n", ii, jj, k, b * sub_n + a * sub_n * sub_n, (ii * n * n * sizeof(double) + jj * n * sizeof(double) + k * sizeof(double)));
            for (kk = k, c = 0; kk < k+sub_n; kk++, c++) {
                result_matrix[c + b * sub_n + a * sub_n * sub_n] = seq_matrix[ii * n * n + jj * n + kk];
            }
        }
    }
}

int nds_tensor_verify(struct resources *res, uint64_t id, uint64_t size, uint64_t sub_size, const double *verify_matrix) {
    double *sub_B, *verify_sub_B;
    size_t i, n, m, k, nn, mm, kk, a, b, c;
    size_t error = 0;
    
    size_t total_iteration;

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

    // initialization
    total_iteration = (M / SUB_M) * (M / SUB_M) * (M / SUB_M);

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

    sub_B = (double *) malloc(SUB_M * SUB_M * SUB_M * sizeof(double) * IO_QUEUE_SZ);
    verify_sub_B = (double *) malloc(SUB_M * SUB_M * SUB_M * sizeof(double));

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
    f_conf.sub_B = sub_B;

    pthread_create(&r_thread_id, NULL, request_thread, &r_conf); 
    pthread_create(&f_thread_id, NULL, fetch_thread, &f_conf);  

    gettimeofday(&h_start, NULL);
    // blockGEMM
    for (n = 0; n < N; n+=SUB_M) {
        for (m = 0; m < M; m+=SUB_M) { 
            for (k = 0; k < K; k+=SUB_M) {
                // memcpy?
                entry = (struct fifo_entry *) fifo_pop(io_queue);
                reassemble_block_tensor_from_seq(verify_matrix, verify_sub_B, M, SUB_M, n, m, k);
                error += verify(verify_sub_B, entry->sub_B, SUB_M * SUB_M * SUB_M);
                fifo_push(complete_queue, entry);
            }
        }
    }

    pthread_join(r_thread_id, NULL); 
    pthread_join(f_thread_id, NULL); 
            
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("Verify duration: %f ms\n", (float) duration / 1000);    
    printf("error: %lu\n", error);
    if (error == 0) {
        printf("TEST PASSED!\n");
    }
    
    free(sub_B);
    free(verify_sub_B);
    fifo_free(io_queue);
    fifo_free(request_queue);
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
    uint64_t matrix_id, n, sub_n;

    int verify_fd, hugepage_fd;
    char *hugepage_addr;
    double *verify_matrix;

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
        printf("usage: %s <matrix_id> <verify_matrix_path> <port>\n", argv[0]);
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

    res.buf = hugepage_addr;
    memset(hugepage_addr, 0, BUF_SIZE);

    printf("hugepage starting address is: %p\n", hugepage_addr);
    printf("socket connection\n");
    rc = make_two_tcp_connection(&res, &config);
    if (rc < 0) {
        perror("sock connect");
        exit(1);
    }


    verify_fd = open(argv[2], O_RDONLY);
    if (verify_fd < 0) {
        perror("open");
        exit(1);
    }

    verify_matrix = (double *) mmap(0, M * N * K * sizeof(double), PROT_READ, MAP_PRIVATE, verify_fd, 0);
    if (verify_matrix==MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    printf("calculating the result of pagerank\n");
    rc = nds_tensor_verify(&res, matrix_id, M, SUB_M, verify_matrix);
    
    close(res.sock);
    close(res.req_sock);
    munmap(hugepage_addr, BUF_SIZE);
    close(hugepage_fd);

    munmap(verify_matrix, M * N * K * sizeof(double));
    close(verify_fd);
    return rc;
}
