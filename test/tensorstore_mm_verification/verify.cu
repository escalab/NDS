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


// ALGO 0: wholeMatrix_Sgemm
// ALGO 1: wholeMatrix_Sgemm_half
// ALGO 2: sequential_blockSgemm
// ALGO 3: sequential_blockSgemm_half
// ALGO 4: tensor_blockSgemm
// ALGO 5: tensor_blockSgemm_half

#ifndef ALGO
#define ALGO 0
#endif

int verify(const float *C, const float *answer, int m, int n) {
    // also need to consider floating point error
    const float relativeTolerance = 1e-3;
    int row, col;
    float relativeError;
    for(row = 0; row < m; ++row) {
        for(col = 0; col < n; ++col) {
            if (isnan(C[row*n + col])) {
                printf("(%d, %d) is NaN\n", row, col);
                return 0; 
            }

            if (isinf(C[row*n + col])) {
                printf("(%d, %d) is inf\n", row, col);
                return 0; 
            }
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
    float *c, *answer_c;
    int n, need_output = 0, is_passed = 0;
    int a_fd, b_fd;
    struct timeval h_start, h_end;
    long duration;

#if ALGO >= 6
    int sub_n;
    int a_tensor_fd, b_tensor_fd;
    double *a_tensor, *b_tensor;
    float *c_reformat;
    if (argc < 5) {
        printf("usage: %s <seq_matrix path> <tensor_matrix path> <matrix size> <submatrix size> [output?]\n", argv[0]);
        exit(1);
    }
    a_tensor_fd = open(argv[2], O_RDONLY);
    b_tensor_fd = open(argv[2], O_RDONLY);
    n = atoi(argv[3]);
    sub_n = atoi(argv[4]);
    if (argc > 5) {
        need_output = atoi(argv[5]);
    }
    a_tensor = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, a_tensor_fd, 0);
    b_tensor = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, b_tensor_fd, 0);
    c_reformat = (float *) calloc(n * n, sizeof(float));

#elif ALGO >= 2
    int sub_n;
    if (argc < 4) {
        printf("usage: %s <matrix path> <matrix size> <submatrix size> [output?]\n", argv[0]);
        exit(1);
    }
    n = atoi(argv[2]);
    sub_n = atoi(argv[3]);
    if (argc > 4) {
        need_output = atoi(argv[4]);
    }
#else
    if (argc < 3) {
        printf("usage: %s <matrix path> <matrix size> [output?]\n", argv[0]);
        exit(1);
    }
    n = atoi(argv[2]);
    if (argc > 3) {
        need_output = atoi(argv[3]);
    }
#endif
    // GEMM configuration.
    a_fd = open(argv[1], O_RDONLY);
    b_fd = open(argv[1], O_RDONLY);

    a = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, a_fd, 0);
    b = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, b_fd, 0);

    c = (float *) calloc(n * n, sizeof(float));
    answer_c = (float *) calloc(n * n, sizeof(float));

    printf("calculating the answer...\n");
    memset(answer_c, 0, n * n * sizeof(float));
    gettimeofday(&h_start, NULL);
    wholeMatrix_Dgemm(n, n, n, a, b, answer_c);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("sequential format GEMM duration: %f ms\n", (float) duration / 1000);    

    printf("calculating the result of the sequential format\n");
    memset(c, 0, n * n * sizeof(float));
    printf("running algorithm %d\n", ALGO);
    gettimeofday(&h_start, NULL);
#if ALGO == 0
    wholeMatrix_Sgemm(n, n, n, a, b, c);
#elif ALGO == 1
    wholeMatrix_Sgemm_half(n, n, n, a, b, c);
#elif ALGO == 2
    sequential_blockDgemm(n, n, n, sub_n, sub_n, sub_n, a, b, c);
#elif ALGO == 3
    sequential_blockDgemm_2D(n, n, n, sub_n, sub_n, sub_n, a, b, c);
#elif ALGO == 4
    sequential_blockSgemm_half_async_v2(n, n, n, sub_n, sub_n, sub_n, a, b, c);
#elif ALGO == 5
    sequential_blockSgemm_half(n, n, n, sub_n, sub_n, sub_n, a, b, c);
#elif ALGO == 6
    tensor_blockSgemm(n, n, n, sub_n, sub_n, sub_n, a_tensor, b_tensor, c);
#elif ALGO == 7
    tensor_blockSgemm_half(n, n, n, sub_n, sub_n, sub_n, a_tensor, b_tensor, c);
#elif ALGO == 8
    tensor_blockSgemm_half_async(n, n, n, sub_n, sub_n, sub_n, a_tensor, b_tensor, c);
#elif ALGO == 9
    tensor_blockSgemm_half_async_v2(n, n, n, sub_n, sub_n, sub_n, a_tensor, b_tensor, c);
#elif ALGO == 10
    tensor_blockSgemm_half_async_v3(n, n, n, sub_n, sub_n, sub_n, a_tensor, b_tensor, c);
#endif
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("GEMM duration: %f ms\n", (float) duration / 1000);    

#if ALGO >= 6
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
    is_passed = verify(c_reformat, answer_c, n, n);
#else
    is_passed = verify(c, answer_c, n, n);
#endif

    if (is_passed && need_output) {
        char filename[64];
        FILE *fptr;
#if ALGO >= 6
        sprintf(filename, "ans_block_%d_%d.bin", n, sub_n);
        printf("writing tensor format answer to %s\n", &filename[0]);
#else
        sprintf(filename, "ans_%d.bin", n);
        printf("writing sequential answer to %s\n", &filename[0]);
#endif
        fptr = fopen(filename, "wb");
        fwrite(c, sizeof(float), n * n, fptr);
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

    munmap(a, sizeof(double) * n * n);
    munmap(b, sizeof(double) * n * n);
    close(a_fd);
    close(b_fd);

    free(answer_c);
    free(c);

#if ALGO >= 6
    munmap(a_tensor, sizeof(double) * n * n);
    munmap(b_tensor, sizeof(double) * n * n);
    close(a_tensor_fd);
    close(b_tensor_fd);
    free(c_reformat);
#endif

    return 0;
}