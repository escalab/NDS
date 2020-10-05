extern "C" {
    #include "rdma.h"
    #include "timing.h"
    #include "fifo.h"
}

#include "cublasGEMM.h"

#define MAX_THREAD 1024
#define IO_QUEUE_SZ (32768UL / 2048UL) 
// #define IO_QUEUE_SZ 1UL

struct fetch_conf {
    struct resources *res;
    uint64_t m, sub_m;
    double *a_sub_d, *b_sub_d;
    char *hugepage_addr;
    struct fifo *queue;
    struct timing_info *fetch_timing;
    struct timing_info *copy_in_timing;
};

struct fifo_entry {
    double *a_sub_d, *b_sub_d;
};

__global__ void d2h_kernel(const double *din, half *dout, size_t dsize) {
	size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
	if (idx < dsize)
	{
		dout[idx] = din[idx];
	}
}

void print_config(struct config_t config);

int verify(const float *C, const float *answer, uint64_t m, uint64_t n) {
    // also need to consider floating point error
    const float relativeTolerance = 1e-3;
    uint64_t row, col;
    float relativeError;
    for(row = 0; row < m; ++row) {
        for(col = 0; col < n; ++col) {
            if (isnan(C[row*n + col]) || isnan(answer[row*n + col])) {
                printf("(%lu, %lu) is NaN\n", row, col);
                return 0; 
            }

            if (isinf(C[row*n + col]) || isinf(answer[row*n + col])) {
                printf("(%lu, %lu) is inf\n", row, col);
                return 0; 
            }
            relativeError = (answer[row*n + col] - C[row*n + col]) / answer[row*n + col];
            if (fabs(relativeError) > relativeTolerance) {
                printf("(%lu, %lu) = %f, supposed to be %f\n", row, col, C[row*n + col], answer[row*n + col]); 
                printf("TEST FAILED\n\n");
                return 0;
            }    
        }
    }
    printf("TEST PASSED\n\n");
    return 1;
}

int cudaMemcpyFromMmap(struct resources *res, void *dst, const void *src, const size_t length) {
    if (sock_write_data(res->sock)) { /* just send a dummy char back and forth */
        fprintf(stderr, "sync error before RDMA ops\n");
        return 1;
    }

    if (sock_read_data(res->sock)) { /* just send a dummy char back and forth */
        fprintf(stderr, "sync error before RDMA ops\n");
        return 1;
    }
    return 0;
}

void *fetch_thread(void *args) {
    struct fetch_conf *conf = (struct fetch_conf *) args;
    uint64_t i, j, k;
    uint64_t dsize = conf->sub_m * conf->sub_m;
    double *ptr_a, *ptr_b;
    char *hugepage_ptr;

    struct fifo_entry *entry = NULL;
    uint64_t count = 0;
    
    for (i = 0; i < conf->m / conf->sub_m; i++) {
        for (j = 0; j < conf->m / conf->sub_m; j++) {
            for (k = 0; k < conf->m / conf->sub_m; k+=IO_QUEUE_SZ) {
                timing_info_push_start(conf->fetch_timing);
                if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
                    fprintf(stderr, "sync error before RDMA ops\n");
                    return NULL;
                }
            
                if (sock_read_data(conf->res->sock)) { /* just send a dummy char back and forth */
                    fprintf(stderr, "sync error before RDMA ops\n");
                    return NULL;
                }
                timing_info_push_end(conf->fetch_timing);

                timing_info_push_start(conf->copy_in_timing);
                for (count = 0; count < IO_QUEUE_SZ; count++) {
                    while (fifo_full(conf->queue)) {

                    }
                    // printf("fetching A[%lu][%lu]\n", i, count+k);
    
                    ptr_a = conf->a_sub_d + dsize * count;
                    hugepage_ptr = conf->hugepage_addr + (count * 2) * dsize * sizeof(double);

                    cudaMemcpy(ptr_a, hugepage_ptr, dsize * sizeof(double), cudaMemcpyHostToDevice);
    
                    // printf("fetching B[%lu][%lu]\n", count+k, j);
                    ptr_b = conf->b_sub_d + dsize * count;
                    hugepage_ptr = conf->hugepage_addr + (count * 2 + 1) * dsize * sizeof(double);
                    
                    cudaMemcpy(ptr_b, hugepage_ptr, dsize * sizeof(double), cudaMemcpyHostToDevice);
    
                    entry = (struct fifo_entry *) malloc(sizeof(struct fifo_entry));
                    entry->a_sub_d = ptr_a;
                    entry->b_sub_d = ptr_b;
                    fifo_push(conf->queue, entry);
                }
                timing_info_push_end(conf->copy_in_timing);
                // if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
                //     fprintf(stderr, "sync error before RDMA ops\n");
                //     return NULL;
                // }
            
                // if (sock_read_data(conf->res->sock)) { /* just send a dummy char back and forth */
                //     fprintf(stderr, "sync error before RDMA ops\n");
                //     return NULL;
                // }
            }
        }
    }
    return NULL;
}

int spdk_nds_blockSgemm_half_pthread(struct resources *res, uint64_t m, uint64_t sub_m, float *c) {
    size_t i, j, k;
    size_t cross_row = m * sub_m, cross_col = sub_m * sub_m;
    double *a_sub_d, *b_sub_d;
    half *a_sub_h, *b_sub_h;
    float *c_sub_f;

    float alpha = 1.0;
    float beta = 1.0;
    
    size_t dsize;
    cublasHandle_t handle;

    struct fifo *queue;
    struct timing_info *queue_timing;
    struct timing_info *fetch_timing;
    struct timing_info *copy_in_timing;
    struct timing_info *d2h_timing;
    struct timing_info *gemm_timing;
    struct timing_info *copy_out_timing;    
    
    pthread_t thread_id; 
    struct fetch_conf conf;

    struct timeval h_start, h_end;
    long duration;

    // initialization
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    dsize = (m / sub_m) * (m / sub_m) * (m / sub_m);
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

    d2h_timing = timing_info_new(dsize);
    if (d2h_timing == NULL) {
        printf("cannot create d2h_timing\n");
        return -1;
    }

    gemm_timing = timing_info_new(dsize);
    if (gemm_timing == NULL) {
        printf("cannot create gemm_timing\n");
        return -1;
    }

    copy_out_timing = timing_info_new(dsize);
    if (copy_out_timing == NULL) {
        printf("cannot create copy_out_timing\n");
        return -1;
    }

    // it causes problem if size == 1
    queue = fifo_new(IO_QUEUE_SZ * 2);
	if (queue == NULL) {
        printf("cannot create queue\n");
        return -1;
	}

    // cuda malloc
    dsize = sub_m * sub_m;
    cudaMalloc((void **) &a_sub_d, sizeof(double) * dsize * IO_QUEUE_SZ);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * dsize * IO_QUEUE_SZ);
    cudaMalloc((void **) &a_sub_h, sizeof(half) * dsize);
    cudaMalloc((void **) &b_sub_h, sizeof(half) * dsize);
    cudaMalloc((void **) &c_sub_f, sizeof(float) * dsize);

    // create thread here
    conf.res = res;
    conf.m = m;
    conf.sub_m = sub_m;
    conf.a_sub_d = a_sub_d;
    conf.b_sub_d = b_sub_d;
    conf.hugepage_addr = res->buf;
    conf.queue = queue;
    conf.fetch_timing = fetch_timing;
    conf.copy_in_timing = copy_in_timing;

    timing_info_set_starting_time(queue_timing);
    timing_info_set_starting_time(fetch_timing);
    timing_info_set_starting_time(copy_in_timing);
    timing_info_set_starting_time(d2h_timing);
    timing_info_set_starting_time(gemm_timing);
    timing_info_set_starting_time(copy_out_timing);

    gettimeofday(&h_start, NULL);
	pthread_create(&thread_id, NULL, fetch_thread, &conf); 
    struct fifo_entry *entry = NULL;
    // blockGEMM
    for (i = 0; i < m / sub_m; i++) {
        for (j = 0; j < m / sub_m; j++) {
            cudaMemset(c_sub_f, 0, sub_m * sub_m * sizeof(float));
            for (k = 0; k < m / sub_m; k++) {
                timing_info_push_start(queue_timing);
                while (fifo_empty(queue)) {

                }
                entry = (struct fifo_entry *) fifo_pop(queue);
                timing_info_push_end(queue_timing);

                timing_info_push_start(d2h_timing);
                d2h_kernel<<<(dsize+MAX_THREAD-1)/MAX_THREAD,MAX_THREAD>>>(entry->a_sub_d, a_sub_h, dsize);
                d2h_kernel<<<(dsize+MAX_THREAD-1)/MAX_THREAD,MAX_THREAD>>>(entry->b_sub_d, b_sub_h, dsize);
                free(entry);
                timing_info_push_end(d2h_timing);

                // gemm
                timing_info_push_start(gemm_timing);
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_m, sub_m, &alpha, b_sub_h, CUDA_R_16F, sub_m, a_sub_h, CUDA_R_16F, sub_m, &beta, c_sub_f, CUDA_R_32F, sub_m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                cudaDeviceSynchronize();
                timing_info_push_end(gemm_timing);
            }
            timing_info_push_start(copy_out_timing);
            cudaMemcpy((c + i * cross_row + j * cross_col), c_sub_f, sub_m * sub_m * sizeof(float), cudaMemcpyDeviceToHost);
            timing_info_push_end(copy_out_timing);
        }
    }
    pthread_join(thread_id, NULL); 

    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("GEMM duration: %f ms\n", (float) duration / 1000);    

    printf("Fetch time: %f ms\n", (float) timing_info_duration(fetch_timing) / 1000);
    printf("Copy in time: %f ms\n", (float) timing_info_duration(copy_in_timing) / 1000);
    printf("queue waiting time: %f ms\n", (float) timing_info_duration(queue_timing) / 1000);
    printf("d2h time: %f ms\n", (float) timing_info_duration(d2h_timing) / 1000);
    printf("GEMM time: %f ms\n", (float) timing_info_duration(gemm_timing) / 1000);
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
    
    tss = timing_info_get_timestamps(queue_timing);
    fptr = fopen("queue_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(queue_timing);

    tss = timing_info_get_timestamps(d2h_timing);
    fptr = fopen("d2h_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(d2h_timing);

    tss = timing_info_get_timestamps(gemm_timing);
    fptr = fopen("gemm_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(gemm_timing);

    tss = timing_info_get_timestamps(copy_out_timing);
    fptr = fopen("copy_out_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(copy_out_timing);

    cublasDestroy(handle);
    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(a_sub_h);
    cudaFree(b_sub_h);
    cudaFree(c_sub_f);
    fifo_free(queue);
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
    double *a, *b;
    int a_fd, b_fd;
    float *c, *answer_c;
    struct timeval h_start, h_end;
    long duration;
    uint64_t n, sub_n;

    int hugepage_fd;
    double *hugepage_addr;

    // RDMA
    struct resources res;
    struct config_t config = {
        "mlx4_0",  /* dev_name */
        NULL,  /* server_name */
        19875, /* tcp_port */
        1,     /* ib_port */
        0     /* gid_idx */
    };

    if (argc < 5) {
        printf("usage: %s <n> <sub_n> <validated matrix> <port>\n", argv[0]);
        exit(1);
    }

    n = (uint64_t) atoll(argv[1]);
    sub_n = (uint64_t) atoll(argv[2]);
    a_fd = open(argv[3], O_RDONLY);
    b_fd = open(argv[3], O_RDONLY);

    config.tcp_port = atoi(argv[4]);

    a = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, a_fd, 0);
    b = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, b_fd, 0);
    
    // GEMM configuration
    c = (float *) calloc(n * n, sizeof(float));
    answer_c = (float *) calloc(n * n, sizeof(float));
    
    /* print the used parameters for info*/
    print_config(config);
    resources_init(&res);

    make_tcp_connection(&res, &config);

    hugepage_fd = open("/dev/hugepages/tensorstore", O_RDWR, 0755);
    if (hugepage_fd < 0) {
        perror("open");
        exit(1);
    }

    hugepage_addr = (double *) mmap(0, BUF_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, hugepage_fd, 0);
    if (hugepage_addr==MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    memset(hugepage_addr, 0, BUF_SIZE);
    memset(c, 0, n * n * sizeof(float));
    memset(answer_c, 0, n * n * sizeof(float));

    printf("hugepage starting address is: %p\n", hugepage_addr);
    res.buf = (char *) hugepage_addr;

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
    rc = spdk_nds_blockSgemm_half_pthread(&res, n, sub_n, c);

    // config_destroy(&config);

    // TODO: load the validation matrix operands
    printf("calculating the answer...\n");
    gettimeofday(&h_start, NULL);
    tensor_blockSgemm(n, n, n, sub_n, sub_n, sub_n, a, b, answer_c);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("Answer GEMM duration: %f ms\n", (float) duration / 1000);    

    verify(c, answer_c, n, n);

    if (resources_destroy(&res)) {
        fprintf(stderr, "failed to destroy resources\n");
        exit(1);
    }

    munmap(hugepage_addr, BUF_SIZE);
    munmap(a, sizeof(double) * n * n);
    munmap(b, sizeof(double) * n * n);
    close(a_fd);
    close(b_fd);

    free(answer_c);
    free(c);

    printf("test result is %i\n", rc);
    return rc;
}
