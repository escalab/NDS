#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cublasGEMM.h"

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// for timing
#include <sys/time.h>

#define THREADS_PER_BLOCK 256
#define ITER_NUM 1

// this one is good because it only takes one element space
int cpu_verify(double *A, double *B, float *C, unsigned int m, unsigned int n, unsigned int k) {
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
                return 0;
            }
        }
    }
    printf("TEST PASSED\n\n");
    return 1;
}

int verify(const float *C, const float *answer, int m, int n) {
    const float relativeTolerance = 1e-3;
    int row, col;
    float relativeError;
    for(row = 0; row < m; ++row) {
        for(col = 0; col < n; ++col) {
            relativeError = (answer[row*n + col] - C[row*n + col]) / answer[row*n + col];
            if (fabs(relativeError) > relativeTolerance) {
                printf("(%d, %d) = %f, supposed to be %f\n", row, col, C[row*n + col], answer[row*n + col]); 
                printf("TEST FAILED\n\n");
                return 0;
            }
        }
    }
    printf("TEST PASSED\n\n");
    return 1;
}


int main(int argc, char** argv) {
    double *a, *b;
    double *a_tensor, *b_tensor;
    float *c, *c_reformat, *answer_c;
    int n, sub_n, need_output = 0, is_passed = 0;
    int a_fd, b_fd;
    int a_tensor_fd, b_tensor_fd;
    struct timeval h_start, h_end;
    long duration;

    if (argc < 5) {
        printf("usage: %s <sequence format path> <tensor format path> <matrix size> <submatrix size> [output?]\n", argv[0]);
        exit(1);
    }

    // GEMM configuration.
    a_fd = open(argv[1], O_RDONLY);
    b_fd = open(argv[1], O_RDONLY);
    a_tensor_fd = open(argv[2], O_RDONLY);
    b_tensor_fd = open(argv[2], O_RDONLY);

    n = atoi(argv[3]);
    sub_n = atoi(argv[4]);

    if (argc > 5) {
        need_output = atoi(argv[5]);
    }

    a = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, a_fd, 0);
    b = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, b_fd, 0);
    a_tensor = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, a_tensor_fd, 0);
    b_tensor = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, b_tensor_fd, 0);
    c_reformat = (float *) calloc(n * n, sizeof(float));

    c = (float *) calloc(n * n, sizeof(float));
    answer_c = (float *) calloc(n * n, sizeof(float));

    printf("calculating the answer...\n");
    for (int i = 0; i < ITER_NUM; i++) {
        memset(answer_c, 0, n * n * sizeof(float));
        gettimeofday(&h_start, NULL);
        wholeMatrixSgemm(n, n, n, a, b, answer_c);
        gettimeofday(&h_end, NULL);
        duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
        printf("sequential format GEMM duration: %f ms\n", (float) duration / 1000);    
    }

    printf("calculating the result of the sequential format\n");
    for (int i = 0; i < ITER_NUM; i++) {
        memset(c, 0, n * n * sizeof(float));
        gettimeofday(&h_start, NULL);
        sequential_blockSgemm(n, n, n, sub_n, sub_n, sub_n, a, b, c);
        gettimeofday(&h_end, NULL);
        duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
        printf("sequential format block-GEMM duration: %f ms\n", (float) duration / 1000);    
    }

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

    is_passed = verify(c, answer_c, n, n);
    if (is_passed && need_output) {
        char filename[64];
        FILE *fptr;
        sprintf(filename, "ans_%d.bin", n);
        fptr = fopen(filename, "wb");
        printf("writing sequential answer to %s\n", filename);
        fwrite(c, sizeof(float), n * n, fptr);
    }

    // GEMM configuration.
    printf("calculating the result of the tensor format\n");
    for (int i = 0; i < ITER_NUM; i++) {
        memset(c, 0, n * n * sizeof(float));
        gettimeofday(&h_start, NULL);
        tensor_blockSgemm(n, n, n, sub_n, sub_n, sub_n, a_tensor, b_tensor, c);
        gettimeofday(&h_end, NULL);
        duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
        printf("tensor format block-GEMM duration: %f ms\n", (float) duration / 1000);
    }
    printf("Reformat from tensor to sequential...\n");
    int count = 0;
    gettimeofday(&h_start, NULL);

    for (int i = 0; i < n; i += sub_n) {
        for (int j = 0; j < n; j += sub_n) {  
            for(int ii = i; ii < i + sub_n; ii++) {
                for(int jj = j; jj < j + sub_n; jj++) {
                    // printf("ii: %d, jj: %d\n", ii, jj);
                    c_reformat[ii * n + jj] = c[count];
                    count++;
                }
            }
        }
    }

    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("Reformat from tensor to sequential duration: %f ms\n", (float) duration / 1000);  
#ifdef DEBUG
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", a_tensor[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", b_tensor[i * n + j]);
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

    is_passed = verify(c_reformat, answer_c, n, n);
    if (is_passed && need_output) {
        char filename[64];
        FILE *fptr;
        sprintf(filename, "ans_block_%d_%d.bin", n, sub_n);
        fptr = fopen(filename, "wb");
        printf("writing sequential answer to %s\n", filename);
        fwrite(c, sizeof(float), n * n, fptr);
    }

    munmap(a, sizeof(double) * n * n);
    munmap(b, sizeof(double) * n * n);
    close(a_fd);
    close(b_fd);

    munmap(a_tensor, sizeof(double) * n * n);
    munmap(b_tensor, sizeof(double) * n * n);
    close(a_tensor_fd);
    close(b_tensor_fd);

    free(answer_c);
    free(c_reformat);
    free(c);
    return 0;
}