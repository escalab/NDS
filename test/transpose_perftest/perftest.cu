#include <stdio.h>
#include <stdlib.h>

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// for timing
#include <sys/time.h>

#include <cublas_v2.h>

#define WARMUP 1
#define ITERATIONS 1

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
        printf("%f ", (float) duration / 1000);
    }
    printf("\n");
    cublasDestroy(handle);
}

int main(int argc, char** argv) {
    double *a_f, *b_f, *a_d, *b_d;
    double *c_d;
    long long n;
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

    a_f = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, a_fd, 0);
    b_f = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, b_fd, 0);

    cudaMalloc((void **) &a_d, sizeof(double) * n * n);
    cudaMalloc((void **) &b_d, sizeof(double) * n * n);
    cudaMalloc((void **) &c_d, sizeof(double) * n * n);

    cudaMemcpy(a_d, a_f, sizeof(double) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_f, sizeof(double) * n * n, cudaMemcpyHostToDevice);

    printf("memory usage: input: %llu bytes, output: %llu bytes\n", sizeof(double) * n * n * 2, sizeof(double) * n * n);
    
    // double
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            test_loop(n, op_arr[i], op_arr[j], a_d, CUDA_R_64F, b_d, CUDA_R_64F, c_d, CUDA_R_64F, CUDA_R_64F);
        }
    }

    // float

    // half


    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    munmap(a_f, sizeof(double) * n * n);
    munmap(b_f, sizeof(double) * n * n);
    close(a_fd);
    close(b_fd);
    return 0;
}