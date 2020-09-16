extern "C" {
    #include "spdkrpc.h"
    #include "timing.h"
    #include "fifo.h"
}

#include <stdio.h>
#include <stdlib.h>
#include "cublasGEMM.h"

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// for timing
#include <sys/time.h>

// for checking nan
#include <math.h>

#include <pthread.h>

#define MAX_THREAD 1024
#define FIFO_QUEUE_SIZE 4

struct fetch_conf {
    struct JSONRPCClient *client;
    int request_id;
    uint64_t m, sub_m;
    double *a_sub_d, *b_sub_d;
    double *hugepage_addr;
    struct fifo *queue;
    struct timing_info *fetch_timing;
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

#ifdef GATHER
int cudaMemcpyFromMmap(struct JSONRPCClient *client, double *dst, const double *src, int id, int x, int y, uint64_t sub_m) {
    size_t return_size; 
    return_size = tensorstore_get_gather_submatrix(client, id, x, y, sub_m);
#else
int cudaMemcpyFromMmap(struct JSONRPCClient *client, double *dst, const double *src, int id, int x, int y) {
    size_t return_size; 
    return_size = tensorstore_get_submatrix(client, id, x, y);
#endif
    if (return_size == 0) {
        return -1;
    }
    cudaMemcpy(dst, src, return_size, cudaMemcpyHostToDevice);
    return 0;
}

void *fetch_thread(void *args) {
    struct fetch_conf *conf = (struct fetch_conf *) args;
    uint64_t i, j, k;
    uint64_t dsize = conf->sub_m * conf->sub_m;
    double *ptr_a, *ptr_b;
    struct fifo_entry *entry = NULL;
    uint64_t count = 0;
    for (i = 0; i < conf->m / conf->sub_m; i++) {
        for (j = 0; j < conf->m / conf->sub_m; j++) {
            for (k = 0; k < conf->m / conf->sub_m; k++) {
                while (fifo_full(conf->queue)) {

                }
                ptr_a = conf->a_sub_d + dsize * (count % FIFO_QUEUE_SIZE);
                ptr_b = conf->b_sub_d + dsize * (count % FIFO_QUEUE_SIZE);
                timing_info_push_start(conf->fetch_timing);
#ifdef GATHER
                cudaMemcpyFromMmap(conf->client, ptr_a, conf->hugepage_addr, conf->request_id, k, i, conf->sub_m);
                cudaMemcpyFromMmap(conf->client, ptr_b, conf->hugepage_addr, conf->request_id, j, k, conf->sub_m);
#else
                cudaMemcpyFromMmap(conf->client, ptr_a, conf->hugepage_addr, conf->request_id, k, i);
                cudaMemcpyFromMmap(conf->client, ptr_b, conf->hugepage_addr, conf->request_id, j, k);
#endif
                timing_info_push_end(conf->fetch_timing);
                count++;

                entry = (struct fifo_entry *) malloc(sizeof(struct fifo_entry));
                entry->a_sub_d = ptr_a;
                entry->b_sub_d = ptr_b;
                fifo_push(conf->queue, entry);
            }
        }
    }
    return NULL;
}

int spdk_nds_blockSgemm_half_pthread(int request_id, uint64_t m, uint64_t sub_m, float *c) {
    size_t i, j, k;
    size_t cross_row = m * sub_m, cross_col = sub_m * sub_m;
    double *hugepage_addr;
    struct JSONRPCClient client;
    int rc;
    double *a_sub_d, *b_sub_d;
    half *a_sub_h, *b_sub_h;
    float *c_sub_f;

    float alpha = 1.0;
    float beta = 1.0;
    struct timeval h_start, h_end;
    unsigned long long fetch_time = 0, gemm_time = 0;
    
    size_t dsize;
    cublasHandle_t handle;

    struct fifo *queue;
    struct timing_info *fetch_timing;
    pthread_t thread_id; 
    struct fetch_conf conf;

    // initialization
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    rc = connect_to_spdkrpc_server(&client);
    if (rc) {
        printf("cannot create conntection to SPDK RPC server");
        return rc;
    }

    hugepage_addr = (double *) mmap_to_tensorstore_hugepage();
    if (hugepage_addr == NULL) {
        return -1;
    }

    dsize = (m / sub_m) * (m / sub_m) * (m / sub_m);
    fetch_timing = timing_info_new(dsize * 2);
    if (fetch_timing == NULL) {
        printf("cannot create fetch_timing\n");
        return -1;
    }

    queue = fifo_new(FIFO_QUEUE_SIZE);
	if (queue == NULL) {
        printf("cannot create queue\n");
        return -1;
	}

    // cuda malloc
    dsize = sub_m * sub_m;
    cudaMalloc((void **) &a_sub_d, sizeof(double) * dsize * FIFO_QUEUE_SIZE);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * dsize * FIFO_QUEUE_SIZE);
    cudaMalloc((void **) &a_sub_h, sizeof(half) * dsize);
    cudaMalloc((void **) &b_sub_h, sizeof(half) * dsize);
    cudaMalloc((void **) &c_sub_f, sizeof(float) * dsize);

    // create thread here
    conf.client = &client;
    conf.request_id = request_id;
    conf.m = m;
    conf.sub_m = sub_m;
    conf.a_sub_d = a_sub_d;
    conf.b_sub_d = b_sub_d;
    conf.hugepage_addr = hugepage_addr;
    conf.queue = queue;
    conf.fetch_timing = fetch_timing;

	pthread_create(&thread_id, NULL, fetch_thread, &conf); 

    struct fifo_entry *entry = NULL;
    // blockGEMM
    for (i = 0; i < m / sub_m; i++) {
        for (j = 0; j < m / sub_m; j++) {
            cudaMemset(c_sub_f, 0, sub_m * sub_m * sizeof(float));
            for (k = 0; k < m / sub_m; k++) {
                // printf("i: %lu, j: %lu, k: %lu\n", i, j, k);
                // memset(hugepage_addr, 0, HUGEPAGE_SZ);
                while (fifo_empty(queue)) {

                }
                entry = (struct fifo_entry *) fifo_pop(queue);

                d2h_kernel<<<(dsize+MAX_THREAD-1)/MAX_THREAD,MAX_THREAD>>>(entry->a_sub_d, a_sub_h, dsize);
                d2h_kernel<<<(dsize+MAX_THREAD-1)/MAX_THREAD,MAX_THREAD>>>(entry->b_sub_d, b_sub_h, dsize);
                
                free(entry);
                // gemm
                gettimeofday(&h_start, NULL);
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_m, sub_m, &alpha, b_sub_h, CUDA_R_16F, sub_m, a_sub_h, CUDA_R_16F, sub_m, &beta, c_sub_f, CUDA_R_32F, sub_m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                cudaDeviceSynchronize();
                gettimeofday(&h_end, NULL);
                gemm_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   
            }
            cudaMemcpy((c + i * cross_row + j * cross_col), c_sub_f, sub_m * sub_m * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }

    pthread_join(thread_id, NULL); 
    printf("data fetch time: %f ms\n", (float) fetch_time / 1000);
    printf("GEMM time: %f ms\n", (float) gemm_time / 1000);

    munmap(hugepage_addr, HUGEPAGE_SZ);
    cublasDestroy(handle);
    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(a_sub_h);
    cudaFree(b_sub_h);
    cudaFree(c_sub_f);
    timing_info_free(fetch_timing);
    fifo_free(queue);
    return 0;
}

int spdk_nds_blockSgemm_half(int request_id, uint64_t m, uint64_t sub_m, float *c) {
    size_t i, j, k;
    size_t cross_row = m * sub_m, cross_col = sub_m * sub_m;
    double *hugepage_addr;
    struct JSONRPCClient client;
    int rc;
    double *a_sub_d, *b_sub_d;
    half *a_sub_h, *b_sub_h;
    float *c_sub_f;

    float alpha = 1.0;
    float beta = 1.0;

    struct timing_info *fetch_timing;
    struct timing_info *d2h_timing;
    struct timing_info *gemm_timing;
    
    size_t dsize;
    cublasHandle_t handle;

    // initialization
    dsize = (m / sub_m) * (m / sub_m) * (m / sub_m);

    fetch_timing = timing_info_new(dsize * 2);
    if (fetch_timing == NULL) {
        printf("cannot create fetch_timing\n");
        return -1;
    }

    d2h_timing = timing_info_new(dsize * 2);
    if (d2h_timing == NULL) {
        printf("cannot create fetch_timing\n");
        return -1;
    }

    gemm_timing = timing_info_new(dsize);
    if (gemm_timing == NULL) {
        printf("cannot create gemm_timing\n");
        return -1;
    }
    
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    rc = connect_to_spdkrpc_server(&client);
    if (rc) {
        printf("cannot create conntection to SPDK RPC server");
        return rc;
    }

    hugepage_addr = (double *) mmap_to_tensorstore_hugepage();
    if (hugepage_addr == NULL) {
        return -1;
    }

    // cuda malloc
    dsize = sub_m * sub_m;
    cudaMalloc((void **) &a_sub_d, sizeof(double) * dsize);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * dsize);
    cudaMalloc((void **) &a_sub_h, sizeof(half) * dsize);
    cudaMalloc((void **) &b_sub_h, sizeof(half) * dsize);
    cudaMalloc((void **) &c_sub_f, sizeof(float) * dsize);

    // blockGEMM
    for (i = 0; i < m / sub_m; i++) {
        for (j = 0; j < m / sub_m; j++) {
            cudaMemset(c_sub_f, 0, sub_m * sub_m * sizeof(float));
            for (k = 0; k < m / sub_m; k++) {
                // printf("i: %lu, j: %lu, k: %lu\n", i, j, k);
                // memset(hugepage_addr, 0, HUGEPAGE_SZ);
                timing_info_push_start(fetch_timing);
#ifdef GATHER
                cudaMemcpyFromMmap(&client, a_sub_d, hugepage_addr, request_id, k, i, sub_m);
                cudaMemcpyFromMmap(&client, b_sub_d, hugepage_addr, request_id, j, k, sub_m);
#else
                cudaMemcpyFromMmap(&client, a_sub_d, hugepage_addr, request_id, k, i);
                cudaMemcpyFromMmap(&client, b_sub_d, hugepage_addr, request_id, j, k);
#endif
                timing_info_push_end(fetch_timing);

                timing_info_push_start(d2h_timing);
                d2h_kernel<<<(dsize+MAX_THREAD-1)/MAX_THREAD,MAX_THREAD>>>(a_sub_d, a_sub_h, dsize);
                d2h_kernel<<<(dsize+MAX_THREAD-1)/MAX_THREAD,MAX_THREAD>>>(b_sub_d, b_sub_h, dsize);
                timing_info_push_end(d2h_timing);

                // gemm
                timing_info_push_start(gemm_timing);
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_m, sub_m, &alpha, b_sub_h, CUDA_R_16F, sub_m, a_sub_h, CUDA_R_16F, sub_m, &beta, c_sub_f, CUDA_R_32F, sub_m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                cudaDeviceSynchronize();
                timing_info_push_end(gemm_timing);
            }
            cudaMemcpy((c + i * cross_row + j * cross_col), c_sub_f, sub_m * sub_m * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
    printf("data fetch time: %f ms\n", (float) timing_info_duration(fetch_timing) / 1000);
    printf("d2h time: %f ms\n", (float) timing_info_duration(d2h_timing) / 1000);
    printf("GEMM time: %f ms\n", (float) timing_info_duration(gemm_timing) / 1000);


    munmap(hugepage_addr, HUGEPAGE_SZ);
    cublasDestroy(handle);
    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(a_sub_h);
    cudaFree(b_sub_h);
    cudaFree(c_sub_f);

    timing_info_free(fetch_timing);
    timing_info_free(d2h_timing);
    timing_info_free(gemm_timing);

    return 0;
}

int spdk_blockSgemm_half(int request_id, uint64_t m, uint64_t sub_m, float *c) {
    uint64_t i, j, k;
    double *hugepage_addr;
    struct JSONRPCClient client;
    int rc;
    double *a_sub_d, *b_sub_d;
    half *a_sub_h, *b_sub_h;
    float *c_sub_f;

    float alpha = 1.0;
    float beta = 1.0;
    struct timeval h_start, h_end;
    unsigned long long fetch_time = 0, gemm_time = 0;
    
    size_t out_pitch, ldc;
    size_t dsize;
    cublasHandle_t handle;

    // initialization
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    rc = connect_to_spdkrpc_server(&client);
    if (rc) {
        printf("cannot create conntection to SPDK RPC server");
        return rc;
    }

    hugepage_addr = (double *) mmap_to_tensorstore_hugepage();
    if (hugepage_addr == NULL) {
        return -1;
    }

    // cuda malloc
    dsize = sub_m * sub_m;
    cudaMalloc((void **) &a_sub_d, sizeof(double) * dsize);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * dsize);
    cudaMalloc((void **) &a_sub_h, sizeof(half) * dsize);
    cudaMalloc((void **) &b_sub_h, sizeof(half) * dsize);

    cudaMallocPitch((void **) &c_sub_f, &out_pitch, sizeof(float) * sub_m, sub_m);
    ldc = out_pitch / sizeof(float);

    // blockGEMM
    for (i = 0; i < m / sub_m; i++) {
        for (j = 0; j < m / sub_m; j++) {
            cudaMemset(c_sub_f, 0, sub_m * sub_m * sizeof(float));
            for (k = 0; k < m / sub_m; k++) {
                // printf("i: %d, j: %d, k: %d\n", i, j, k);
                // memset(hugepage_addr, 0, HUGEPAGE_SZ);
                gettimeofday(&h_start, NULL);
#ifdef GATHER
                cudaMemcpyFromMmap(&client, a_sub_d, hugepage_addr, request_id, k, i, sub_m);
                cudaMemcpyFromMmap(&client, b_sub_d, hugepage_addr, request_id, j, k, sub_m);
#else
                cudaMemcpyFromMmap(&client, a_sub_d, hugepage_addr, request_id, k, i);
                cudaMemcpyFromMmap(&client, b_sub_d, hugepage_addr, request_id, j, k);
#endif
                gettimeofday(&h_end, NULL);
                fetch_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   
                d2h_kernel<<<(dsize+MAX_THREAD-1)/MAX_THREAD,MAX_THREAD>>>(a_sub_d, a_sub_h, dsize);
                d2h_kernel<<<(dsize+MAX_THREAD-1)/MAX_THREAD,MAX_THREAD>>>(b_sub_d, b_sub_h, dsize);
                // gemm
                gettimeofday(&h_start, NULL);
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_m, sub_m, &alpha, b_sub_h, CUDA_R_16F, ldc, a_sub_h, CUDA_R_16F, ldc, &beta, c_sub_f, CUDA_R_32F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                cudaDeviceSynchronize();
                gettimeofday(&h_end, NULL);
                gemm_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   
            }
            cudaMemcpy2D((c + (i * sub_m) * m + (j * sub_m)), m * sizeof(float), c_sub_f, out_pitch, sizeof(float) * sub_m, sub_m, cudaMemcpyDeviceToHost);
        }
    }
    printf("data fetch time: %f ms\n", (float) fetch_time / 1000);
    printf("GEMM time: %f ms\n", (float) gemm_time / 1000);

    munmap(hugepage_addr, HUGEPAGE_SZ);
    cublasDestroy(handle);
    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(a_sub_h);
    cudaFree(b_sub_h);
    cudaFree(c_sub_f);
    return 0;
}

int spdk_nds_blockSgemm_half_alt(int request_id, uint64_t m, uint64_t sub_m, float *c) {
    uint64_t i, j, k;
    size_t cross_row = m * sub_m, cross_col = sub_m * sub_m;
    double *hugepage_addr;
    struct JSONRPCClient client;
    int rc;
    double *a_sub_d, *b_sub_d;
    half *a_sub_h, *b_sub_h;
    float *c_sub_f;

    struct timing_info *fetch_timing;
    struct timing_info *d2h_timing;
    struct timing_info *gemm_timing;

    float alpha = 1.0;
    float beta = 1.0;
    
    size_t dsize;
    cublasHandle_t handle;

    dsize = (m / sub_m) * (m / sub_m) * (m / sub_m);

    fetch_timing = timing_info_new(dsize * 2);
    if (fetch_timing == NULL) {
        printf("cannot create fetch_timing\n");
        return -1;
    }

    d2h_timing = timing_info_new(dsize * 2);
    if (d2h_timing == NULL) {
        printf("cannot create fetch_timing\n");
        return -1;
    }

    gemm_timing = timing_info_new(dsize);
    if (gemm_timing == NULL) {
        printf("cannot create gemm_timing\n");
        return -1;
    }

    // initialization
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    rc = connect_to_spdkrpc_server(&client);
    if (rc) {
        printf("cannot create conntection to SPDK RPC server\n");
        return rc;
    }

    hugepage_addr = (double *) mmap_to_tensorstore_hugepage();
    if (hugepage_addr == NULL) {
        return -1;
    }

    // cuda malloc
    dsize = sub_m * sub_m;
    cudaMalloc((void **) &a_sub_d, sizeof(double) * dsize);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * dsize);
    cudaMalloc((void **) &a_sub_h, sizeof(half) * dsize);
    cudaMalloc((void **) &b_sub_h, sizeof(half) * dsize);

    cudaMalloc((void **) &c_sub_f, sizeof(float) * dsize);
    cudaMemset(c_sub_f, 0, sizeof(float) * dsize);

    // blockGEMM
    for (k = 0; k < m / sub_m; k++) {
        for (i = 0; i < m / sub_m; i++) {
            timing_info_push_start(fetch_timing);
#ifdef GATHER
            cudaMemcpyFromMmap(&client, a_sub_d, hugepage_addr, request_id, k, i, sub_m);
#else
            cudaMemcpyFromMmap(&client, a_sub_d, hugepage_addr, request_id, k, i);
#endif
            timing_info_push_end(fetch_timing);

            timing_info_push_start(d2h_timing);
            d2h_kernel<<<(dsize+MAX_THREAD-1)/MAX_THREAD,MAX_THREAD>>>(a_sub_d, a_sub_h, dsize); 
            timing_info_push_end(d2h_timing);

            for (j = 0; j < m / sub_m; j++) {
                // memset(hugepage_addr, 0, HUGEPAGE_SZ);
                // printf("i: %d, j: %d, k: %d\n", i, j, k);
                timing_info_push_start(fetch_timing);
#ifdef GATHER
                cudaMemcpyFromMmap(&client, b_sub_d, hugepage_addr, request_id, j, k, sub_m);
#else
                cudaMemcpyFromMmap(&client, b_sub_d, hugepage_addr, request_id, j, k);
#endif
                timing_info_push_end(fetch_timing);

                timing_info_push_start(d2h_timing);
                d2h_kernel<<<(dsize+MAX_THREAD-1)/MAX_THREAD,MAX_THREAD>>>(b_sub_d, b_sub_h, dsize);
                timing_info_push_end(d2h_timing);
                // gemm
                cudaMemcpy(c_sub_f, (c + i * cross_row + j * cross_col), dsize * sizeof(float), cudaMemcpyHostToDevice);                
                timing_info_push_start(gemm_timing);
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_m, sub_m, &alpha, b_sub_h, CUDA_R_16F, sub_m, a_sub_h, CUDA_R_16F, sub_m, &beta, c_sub_f, CUDA_R_32F, sub_m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                cudaDeviceSynchronize();
                timing_info_push_end(gemm_timing);
                cudaMemcpy((c + i * cross_row + j * cross_col), c_sub_f, dsize * sizeof(float), cudaMemcpyDeviceToHost);
            }
        }   
    }
    printf("data fetch time: %f ms\n", (float) timing_info_duration(fetch_timing) / 1000);
    printf("d2h time: %f ms\n", (float) timing_info_duration(d2h_timing) / 1000);
    printf("GEMM time: %f ms\n", (float) timing_info_duration(gemm_timing) / 1000);

    munmap(hugepage_addr, HUGEPAGE_SZ);
    cublasDestroy(handle);
    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(a_sub_h);
    cudaFree(b_sub_h);
    cudaFree(c_sub_f);
    timing_info_free(fetch_timing);
    timing_info_free(d2h_timing);
    timing_info_free(gemm_timing);
    return 0;
}


int main(int argc, char** argv) {
    double *a, *b;
    int a_fd, b_fd;
    float *c, *answer_c;
    struct timeval h_start, h_end;
    long duration;
    int request_id;
    uint64_t n, sub_n;
    
    if (argc < 5) {
        printf("usage: %s <req matrix id> <n> <sub_n> <validated matrix>\n", argv[0]);
        exit(1);
    }

    request_id = atoi(argv[1]);
    n = (uint64_t) atoll(argv[2]);
    sub_n = (uint64_t) atoll(argv[3]);

    a_fd = open(argv[4], O_RDONLY);
    b_fd = open(argv[4], O_RDONLY);

    a = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, a_fd, 0);
    b = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, b_fd, 0);

    // GEMM configuration
    c = (float *) calloc(n * n, sizeof(float));
    answer_c = (float *) calloc(n * n, sizeof(float));

    // TODO: load the validation matrix operands
    printf("calculating the answer...\n");
    memset(answer_c, 0, n * n * sizeof(float));
    gettimeofday(&h_start, NULL);
    tensor_blockSgemm(n, n, n, sub_n, sub_n, sub_n, a, b, answer_c);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("Answer GEMM duration: %f ms\n", (float) duration / 1000);    

    printf("calculating the result of SPDK GEMM\n");
    memset(c, 0, n * n * sizeof(float));
    gettimeofday(&h_start, NULL);
    spdk_nds_blockSgemm_half(request_id, n, sub_n, c);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("GEMM duration: %f ms\n", (float) duration / 1000);    

    verify(c, answer_c, n, n);

    // if (is_passed && need_output) {
    //     char filename[64];
    //     FILE *fptr;
    //     sprintf(filename, "ans_%d.bin", n);
    //     printf("writing sequential answer to %s\n", &filename[0]);
    //     fptr = fopen(filename, "wb");
    //     fwrite(c, sizeof(float), n * n, fptr);
    // }

#ifdef DEBUG
    uint64_t i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", a[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", b[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", c[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
#endif

    munmap(a, sizeof(double) * n * n);
    munmap(b, sizeof(double) * n * n);
    close(a_fd);
    close(b_fd);

    free(answer_c);
    free(c);

    return 0;
}