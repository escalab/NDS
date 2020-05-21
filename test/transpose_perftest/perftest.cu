#include <stdio.h>
#include <stdlib.h>

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// for timing
#include <sys/time.h>

#include <cublas_v2.h>

#define THREADS_PER_BLOCK 256
#define WARMUP 1
#define ITERATIONS 10

__global__ void d2f_kernel(const double *din, float *dout, size_t dsize) {
	size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
	if (idx < dsize)
	{
		dout[idx] = din[idx];
	}
}

__global__ void f2h_kernel(const float *din, half *dout, size_t dsize) {
	size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
	if (idx < dsize)
	{
		dout[idx] = din[idx];
	}
}

int isfinite_arr(double *ptr, size_t n) {
    size_t i;

    for (i = 0; i < n; i++) {
        if (!isfinite(ptr[i])) {
            return 0; 
        }
    }
    return 1;
}

int isfinite_arr(float *ptr, size_t n) {
    size_t i;
    for (i = 0; i < n; i++) {
        if (!isfinite(ptr[i])) {
            return 0; 
        }
    }
    return 1;
}

void test_loop(size_t n, cublasOperation_t a_op, cublasOperation_t b_op, 
    void *a, cudaDataType_t a_type, void *b, cudaDataType_t b_type,  
    void *c, cudaDataType_t c_type, cudaDataType_t compute_type) {
    int i;
    float alpha = 1.0;
    float beta = 0.0;

    struct timeval h_start, h_end;
    long duration;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    for (i = 0; i < WARMUP; i++) {
        cublasGemmEx(handle, a_op, b_op, n, n, n, &alpha, a, a_type, n, b, b_type, n, &beta, c, c_type, n, compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
    }

    for (i = 0; i < ITERATIONS; i++) {
        gettimeofday(&h_start, NULL);
        cublasGemmEx(handle, a_op, b_op, n, n, n, &alpha, a, a_type, n, b, b_type, n, &beta, c, c_type, n, compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
        gettimeofday(&h_end, NULL);
        duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
        printf("%.3f ", (float) duration / 1000);
    }
    printf("\n");
    cublasDestroy(handle);
}

int main(int argc, char** argv) {
    double *a, *b;
    double *a_d, *b_d, *c_d, *c_host_d;
    float *a_f, *b_f, *c_f, *c_host_f;
    half *a_h, *b_h;
    float *c_h, *c_host_h;
    size_t n, dsize;
    int a_fd, b_fd;
    int i, j;
    cublasOperation_t op_arr[] = {CUBLAS_OP_N, CUBLAS_OP_T};

    if (argc < 4) {
        printf("usage: %s <matrix A path> <matrix B path> <matrix size>\n", argv[0]);
        exit(1);
    }

    // GEMM configuration.
    a_fd = open(argv[1], O_RDONLY);
    b_fd = open(argv[2], O_RDONLY);
    n = atoi(argv[3]);
    dsize = n * n;

    a = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, a_fd, 0);
    b = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, b_fd, 0);
    cudaMalloc((void **) &a_d, sizeof(double) * n * n);
    cudaMalloc((void **) &b_d, sizeof(double) * n * n);
    cudaMalloc((void **) &c_d, sizeof(double) * n * n);

    cudaMemcpy(a_d, a, sizeof(double) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(double) * n * n, cudaMemcpyHostToDevice);

    munmap(a, sizeof(double) * n * n);
    munmap(b, sizeof(double) * n * n);
    close(a_fd);
    close(b_fd);

    printf("memory usage: input: %lu bytes, output: %lu bytes\n", sizeof(double) * n * n * 2, sizeof(double) * n * n);
    
    // double
    c_host_d = (double *) malloc(sizeof(double) * n * n);
    printf("Running double precision...\n");
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            test_loop(n, op_arr[i], op_arr[j], a_d, CUDA_R_64F, b_d, CUDA_R_64F, c_d, CUDA_R_64F, CUDA_R_64F);
            cudaMemcpy(c_host_d, c_d, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
            if (!isfinite_arr(c_host_d, dsize)) {
                printf("found non-finite result, abort\n");
                exit(-1);
            }
        }
    }
    free(c_host_d);

    // float
    cudaFree(c_d);
    cudaMalloc((void **) &a_f, sizeof(float) * n * n);
    d2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(a_d, a_f, dsize);
    cudaFree(a_d);
    cudaMalloc((void **) &b_f, sizeof(float) * n * n);
    d2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(b_d, b_f, dsize);
    cudaFree(b_d);
    cudaMalloc((void **) &c_f, sizeof(float) * n * n);

    printf("Running float precision...\n");
    c_host_f = (float *) malloc(sizeof(float) * n * n);
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            test_loop(n, op_arr[i], op_arr[j], a_f, CUDA_R_32F, b_f, CUDA_R_32F, c_f, CUDA_R_32F, CUDA_R_32F);
            cudaMemcpy(c_host_f, c_f, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
            if (!isfinite_arr(c_host_f, dsize)) {
                printf("found non-finite result, abort\n");
                exit(-1);
            }
        }
    }
    free(c_host_f);

    // half
    cudaFree(c_f);
    cudaMalloc((void **) &a_h, sizeof(half) * n * n);
    f2h_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(a_f, a_h, dsize);
    cudaFree(a_f);
    cudaMalloc((void **) &b_h, sizeof(half) * n * n);
    f2h_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(b_f, b_h, dsize);
    cudaFree(b_f);
    cudaMalloc((void **) &c_h, sizeof(float) * n * n);

    printf("Running half precision...\n");
    c_host_h = (float *) malloc(sizeof(float) * n * n);
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            test_loop(n, op_arr[i], op_arr[j], a_h, CUDA_R_16F, b_h, CUDA_R_16F, c_h, CUDA_R_32F, CUDA_R_32F);
            cudaMemcpy(c_host_h, c_h, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
            if (!isfinite_arr(c_host_h, dsize)) {
                printf("found non-finite result, abort\n");
                exit(-1);
            }
        }
    }
    free(c_host_h);
    cudaFree(a_h);
    cudaFree(b_h);
    cudaFree(c_h);

    return 0;
}