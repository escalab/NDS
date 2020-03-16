#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

void verify(double *A, double *B, float *C, unsigned int m, unsigned int n, unsigned int k) {
    const float relativeTolerance = 1e-3;
  
    for(int row = 0; row < m; ++row) {
        for(int col = 0; col < n; ++col) {
            float sum = 0;
            for(unsigned int i = 0; i < k; ++i) {
                // printf("C[%u][%u] = A[%u][%u] * B[%u][%u]\n", row, col, row, i, i, col);
                sum += (float) A[row*k + i]*B[i*n + col];
            }
            float relativeError = (sum - C[row*n + col])/sum;
            if (fabs(relativeError) > relativeTolerance) {
                printf("(%d, %d) = %f, supposed to be %f\n", row, col, C[row*n + col], sum); 
                printf("TEST FAILED\n\n");
                exit(0);
            }
        }
    }
    printf("TEST PASSED\n\n");
}

float* doMultiply2Matrices(int m, int n, int k, double *a, double *b, float *c) {
    double alpha = 1.0;
    double beta = 0.0;
    double *a_d, *b_d, *c_d, *c_h;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMalloc((void **) &a_d, sizeof(double) * m * k);
    cudaMalloc((void **) &b_d, sizeof(double) * k * n);
    cudaMalloc((void **) &c_d, sizeof(double) * m * n);
    c_h = (double *) malloc(sizeof(double) * m * n);

    cudaMemcpy(a_d, a, sizeof(double) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(double) * k * n, cudaMemcpyHostToDevice);

    // the memory layout is different from we know
    // a = [0 1; b = [3 2; 
    //      2 3]      1 0]
    // if use a_d then b_d, c[0][0] will be a[0, 0] * b[0, 0] + a[1, 0] * b[0, 1] = 4
    // with b_d then a_d, c[0][0] will be a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] = 1
    // maybe that's because inside GPU it uses column major storage.
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, b_d, k, a_d, m, &beta, c_d, m);
    
    cudaMemcpy(c_h, c_d, sizeof(double) * m * n, cudaMemcpyDeviceToHost);
    cublasDestroy(handle);

    for (int i = 0; i < m * n; ++i) {
        c[i] = (float) c_h[i];
    }

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(c_h);
    return c;
}


int main(int argc, char** argv) {
    double *a, *b;
    float *c;
    int n;
    int a_fd, b_fd;

    if (argc < 3) {
        printf("usage: %s <matrix path> <matrix size>\n", argv[0]);
        exit(1);
    }

    // GEMM configuration.
    a_fd = open(argv[1], O_RDONLY);
    b_fd = open(argv[1], O_RDONLY);

    n = atoi(argv[2]);
    a = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, a_fd, 0);
    b = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, b_fd, 0);
  
    // a = (double *) malloc(sizeof(double) * n * n);
    // b = (double *) malloc(sizeof(double) * n * n);

    c = (float *) calloc(n * n, sizeof(float));

    doMultiply2Matrices(n, n, n, a, b, c);

#ifdef DEBUG
    int i, j;
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
    verify(a, b, c, n, n, n);
    munmap(a, sizeof(double) * n * n);
    munmap(b, sizeof(double) * n * n);
    close(a_fd);
    close(b_fd);
    free(c);
    return 0;
}