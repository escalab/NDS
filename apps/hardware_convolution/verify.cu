extern "C" {
    #include "rdma.h"
    #include "timing.h"
    #include "fifo.h"
}
// CUDA runtime
#include <cuda_runtime.h>

#include "convolutionSeparable_common.h"

#define MAX_THREAD 1024

#define HUGEPAGE_SZ (4UL * 1024UL * 1024UL * 1024UL)
#define M 65536UL
#define SUB_M 4096UL
#define AGGREGATED_SZ (SUB_M * SUB_M * 8UL)

// #define IO_QUEUE_SZ (HUGEPAGE_SZ / AGGREGATED_SZ)
#define IO_QUEUE_SZ 1UL

void print_config(struct config_t config);

struct fetch_conf {
    struct resources *res;
    uint64_t m, sub_m;
    double *a_sub_d, *b_sub_d;
    char *hugepage_addr;
    struct fifo *sending_queue;
    struct fifo *complete_queue;
    struct timing_info *fetch_timing;
    struct timing_info *copy_in_timing;
};

struct request_conf {
    struct resources *res;
    uint64_t id;
    uint64_t m, sub_m;
};

struct fifo_entry {
    struct response *resp;
};

int verify(const double *C, uint64_t m, uint64_t n) {
    // also need to consider floating point error
    uint64_t row, col;
    int rc = 0;
    for(row = 0; row < m; ++row) {
        for(col = 0; col < n; ++col) {
            if (isnan(C[row*n + col])) {
                printf("(%lu, %lu) is NaN\n", row, col);
                rc = 1; 
                return rc;
            }

            if (isinf(C[row*n + col])) {
                printf("(%lu, %lu) is inf\n", row, col);
                rc = 1; 
                return rc;
            }
        }
    }
    printf("TEST PASSED\n\n");
    return rc;
}

int cudaMemcpyFromMmap(struct fetch_conf *conf, char *dst, const char *src, const size_t length) {
    struct response *res = NULL;

    timing_info_push_start(conf->fetch_timing);
    res = sock_read_offset(conf->res->sock);
    if (res == NULL) {
        fprintf(stderr, "sync error before RDMA ops\n");
        return 1;
    }
    // if (res->id == 0) {
    //     printf("fetching A[%lu][%lu]\n", res->y, res->x);
    // } else {
    //     printf("fetching B[%lu][%lu]\n", res->y, res->x);
    // }

    timing_info_push_end(conf->fetch_timing);

    timing_info_push_start(conf->copy_in_timing);
    cudaMemcpy(dst, src + res->offset, length, cudaMemcpyHostToDevice);
    timing_info_push_end(conf->copy_in_timing);

    free(res);
    if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
        fprintf(stderr, "sync error before RDMA ops\n");
        return 1;
    }
    // if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
    //     fprintf(stderr, "sync error before RDMA ops\n");
    //     return 1;
    // }
    return 0;
}

void *fetch_thread(void *args) {
    struct fetch_conf *conf = (struct fetch_conf *) args;
    uint64_t i, j, k;
    struct fifo_entry *entry = NULL;

    for (i = 0; i < conf->m / conf->sub_m; i++) {
        for (j = 0; j < conf->m / conf->sub_m; j++) {
            entry = (struct fifo_entry *) fifo_pop(conf->complete_queue);

            timing_info_push_start(conf->fetch_timing);
            entry->resp = sock_read_offset(conf->res->sock);
            if (entry->resp == NULL) {
                fprintf(stderr, "sync error before RDMA ops\n");
                return NULL;
            }

            if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
                fprintf(stderr, "sync error before RDMA ops\n");
                return NULL;
            }
            timing_info_push_end(conf->fetch_timing);                
            fifo_push(conf->sending_queue, entry);
        }
    }
    return NULL;
}

void *request_thread(void *args) {
    struct request_conf *conf = (struct request_conf *) args;
    uint64_t i, j;
    for (i = 0; i < conf->m / conf->sub_m; i++) {
        for (j = 0; j < conf->m / conf->sub_m; j++) {
            sock_write_request(conf->res->req_sock, conf->id, j, i, SUB_M, 1, 0);
            sock_read_data(conf->res->req_sock);
        }
    }
    return NULL;
}

int spdk_nds_blockSgemm_half_pthread(struct resources *res, uint64_t id, uint64_t m, uint64_t sub_m, double *c) {
    size_t i, j;

    double *h_Kernel;
    double *d_Input, *d_Output, *d_Buffer;
    size_t dsize;

    struct fifo *sending_queue;
    struct fifo *complete_queue; 
    struct fifo_entry *entries = (struct fifo_entry *) calloc(IO_QUEUE_SZ, sizeof(struct fifo_entry));
    struct fifo_entry *entry = NULL;

    struct timing_info *fetch_timing;
    struct timing_info *copy_in_timing;
    struct timing_info *memset_timing;
    struct timing_info *queue_timing;
    struct timing_info *convolution_row_timing;
    struct timing_info *convolution_col_timing;
    struct timing_info *copy_out_timing;    
    
    pthread_t f_thread_id; 
    struct fetch_conf f_conf;

    pthread_t r_thread_id; 
    struct request_conf r_conf;

    struct timeval h_start, h_end;
    long duration;

    // initialization
    dsize = (m / sub_m) * (m / sub_m);
    queue_timing = timing_info_new(dsize);
    if (queue_timing == NULL) {
        printf("cannot create queue_timing\n");
        return -1;
    }

    fetch_timing = timing_info_new(dsize);
    if (fetch_timing == NULL) {
        printf("cannot create fetch_timing\n");
        return -1;
    }

    copy_in_timing = timing_info_new(dsize);
    if (copy_in_timing == NULL) {
        printf("cannot create copy_in_timing\n");
        return -1;
    }

    memset_timing = timing_info_new(dsize);
    if (memset_timing == NULL) {
        printf("cannot create memset_timing\n");
        return -1;
    }

    convolution_row_timing = timing_info_new(dsize);
    if (convolution_row_timing == NULL) {
        printf("cannot create convolution_row_timing\n");
        return -1;
    }

    convolution_col_timing = timing_info_new(dsize);
    if (convolution_col_timing == NULL) {
        printf("cannot create convolution_col_timing\n");
        return -1;
    }

    copy_out_timing = timing_info_new(dsize);
    if (copy_out_timing == NULL) {
        printf("cannot create copy_out_timing\n");
        return -1;
    }

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

    // cuda malloc
    dsize = sub_m * sub_m;

    printf("Image Width x Height = %lu x %lu\n\n", sub_m, sub_m);
    printf("Allocating and initializing host arrays...\n");
    h_Kernel    = (double *)malloc(KERNEL_LENGTH * sizeof(double));
    srand(5);

    for (unsigned int i = 0; i < KERNEL_LENGTH; i++)
    {
        h_Kernel[i] = (double)(rand() % 16);
    }

    printf("Allocating and initializing CUDA arrays...\n");
    cudaMalloc((void **)&d_Input,   dsize * sizeof(double));
    cudaMalloc((void **)&d_Output,  dsize * sizeof(double));
    cudaMalloc((void **)&d_Buffer , dsize * sizeof(double));

    setConvolutionKernel(h_Kernel);

    r_conf.res = res;
    r_conf.id = id;
    r_conf.m = M;
    r_conf.sub_m = SUB_M;
    pthread_create(&r_thread_id, NULL, request_thread, &r_conf); 

    // create thread here
    f_conf.res = res;
    f_conf.m = m;
    f_conf.sub_m = sub_m;
    f_conf.hugepage_addr = res->buf;
    f_conf.sending_queue = sending_queue;
    f_conf.complete_queue = complete_queue;
    f_conf.fetch_timing = fetch_timing;
    f_conf.copy_in_timing = copy_in_timing;

    timing_info_set_starting_time(queue_timing);
    timing_info_set_starting_time(memset_timing);
    timing_info_set_starting_time(fetch_timing);
    timing_info_set_starting_time(copy_in_timing);
    timing_info_set_starting_time(convolution_row_timing);
    timing_info_set_starting_time(convolution_col_timing);
    timing_info_set_starting_time(copy_out_timing);
	pthread_create(&f_thread_id, NULL, fetch_thread, &f_conf); 

    gettimeofday(&h_start, NULL);
    
    // blockGEMM
    for (i = 0; i < m / sub_m; i++) {
        for (j = 0; j < m / sub_m; j++) {
            timing_info_push_start(memset_timing);
            cudaMemset(d_Output, 0, dsize * sizeof(double));
            timing_info_push_end(memset_timing);

            timing_info_push_start(queue_timing);
            entry = (struct fifo_entry *) fifo_pop(sending_queue);
            timing_info_push_end(queue_timing);

            timing_info_push_start(copy_in_timing);
            cudaMemcpy(d_Input, res->buf + entry->resp->offset, dsize * sizeof(double), cudaMemcpyHostToDevice);
            free(entry->resp);
            fifo_push(complete_queue, entry);
            timing_info_push_end(copy_in_timing);

            timing_info_push_start(convolution_row_timing);
            convolutionRowsGPU(
                d_Buffer,
                d_Input,
                sub_m,
                sub_m
            );
            timing_info_push_end(convolution_row_timing);
    
            timing_info_push_start(convolution_col_timing);
            convolutionColumnsGPU(
                d_Output,
                d_Buffer,
                sub_m,
                sub_m
            );
            cudaDeviceSynchronize();
            timing_info_push_end(convolution_col_timing);

            timing_info_push_start(copy_out_timing);
            cudaMemcpy(c, d_Output, dsize * sizeof(double), cudaMemcpyDeviceToHost);
            timing_info_push_end(copy_out_timing);
        }
    }

    sock_write_request(res->req_sock, -1, 0, 0, SUB_M, 1, 0);
    sock_read_data(res->req_sock);
    pthread_join(r_thread_id, NULL); 
    pthread_join(f_thread_id, NULL); 

    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("End-to-end duration: %f ms\n", (float) duration / 1000);    

    printf("Fetch time: %f ms\n", (float) timing_info_duration(fetch_timing) / 1000);
    printf("Copy in time: %f ms\n", (float) timing_info_duration(copy_in_timing) / 1000);
    printf("memset time: %f ms\n", (float) timing_info_duration(memset_timing) / 1000);
    printf("queue waiting time: %f ms\n", (float) timing_info_duration(queue_timing) / 1000);
    printf("convolution row time: %f ms\n", (float) timing_info_duration(convolution_row_timing) / 1000);
    printf("convolution col time time: %f ms\n", (float) timing_info_duration(convolution_col_timing) / 1000);
    printf("copy out time: %f ms\n", (float) timing_info_duration(copy_out_timing) / 1000);

    struct timestamps *tss = NULL;
    FILE *fptr;
    tss = timing_info_get_timestamps(fetch_timing);
    fptr = fopen("fetch_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(fetch_timing);

    tss = timing_info_get_timestamps(copy_in_timing);
    fptr = fopen("copy_in_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(copy_in_timing);

    tss = timing_info_get_timestamps(memset_timing);
    fptr = fopen("memset_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(memset_timing);
    
    tss = timing_info_get_timestamps(queue_timing);
    fptr = fopen("queue_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(queue_timing);

    tss = timing_info_get_timestamps(convolution_row_timing);
    fptr = fopen("convolution_row_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(convolution_row_timing);

    tss = timing_info_get_timestamps(convolution_col_timing);
    fptr = fopen("convolution_col_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(convolution_col_timing);

    tss = timing_info_get_timestamps(copy_out_timing);
    fptr = fopen("copy_out_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(copy_out_timing);

    cudaFree(d_Buffer);
    cudaFree(d_Output);
    cudaFree(d_Input);
    free(h_Kernel);

    fifo_free(sending_queue);
    fifo_free(complete_queue);
    free(entries);    
    cudaDeviceReset();

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
    double *c;
    uint64_t matrix_id, n, sub_n;

    int hugepage_fd;
    char *hugepage_addr;

    // RDMA
    struct resources res;
    struct config_t config = {
        "mlx4_0",  /* dev_name */
        "192.168.1.10",  /* server_name */
        19875, /* tcp_port */
        1,     /* ib_port */
        0     /* gid_idx */
    };

    if (argc < 5) {
        printf("usage: %s <matrix_id> <n> <sub_n> <port>\n", argv[0]);
        exit(1);
    }
    matrix_id = (uint64_t) atoll(argv[1]);
    n = (uint64_t) atoll(argv[2]);
    sub_n = (uint64_t) atoll(argv[3]);

    config.tcp_port = atoi(argv[4]);
    
    // GEMM configuration
    c = (double *) calloc(sub_n * sub_n, sizeof(double));
    
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
    memset(c, 0, sub_n * sub_n * sizeof(double));

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

    fprintf(stdout, "running server\n");

    printf("calculating the result of SPDK GEMM\n");
    rc = spdk_nds_blockSgemm_half_pthread(&res, matrix_id, n, sub_n, c);

    // config_destroy(&config);

    munmap(hugepage_addr, BUF_SIZE);
    if (resources_destroy(&res)) {
        fprintf(stderr, "failed to destroy resources\n");
        exit(1);
    }
    close(res.req_sock);

    for (int i = 0; i < 16; i++) {
        printf("%f ", c[i]);
    }
    printf("\n");
    rc = verify(c, sub_n, sub_n);
    free(c);

    printf("test result is %i\n", rc);
    return rc;
}
