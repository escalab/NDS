#include <stdio.h>
#include <stdlib.h>

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// for timing
#include <sys/time.h>

#include <cublas_v2.h>


int main(int argc, char** argv) {
    double *a_f, *b_f, *a_d, *b_d;
    double *c, *c_d;
    long long n;
    int a_fd, b_fd;
    struct timeval h_start, h_end;
    long duration;
    float alpha = 1.0;
    float beta = 0.0;
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

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
    c = (double *) calloc(n * n, sizeof(double));

    cudaMalloc((void **) &a_d, sizeof(double) * n * n);
    cudaMalloc((void **) &b_d, sizeof(double) * n * n);
    cudaMalloc((void **) &c_d, sizeof(double) * n * n);

    printf("memory usage: input: %llu bytes, output: %llu bytes\n", sizeof(double) * n * n * 2, sizeof(double) * n * n);
    
    printf("calculating the answer...\n");
    gettimeofday(&h_start, NULL);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, b_d, CUDA_R_64F, n, a_d, CUDA_R_64F, n, &beta, c_d, CUDA_R_64F, n, CUDA_R_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("computation duration: %f ms\n", (float) duration / 1000);    

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    
    munmap(a_f, sizeof(double) * n * n);
    munmap(b_f, sizeof(double) * n * n);
    close(a_fd);
    close(b_fd);
    free(c);
    return 0;
}