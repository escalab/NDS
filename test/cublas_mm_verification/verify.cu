#include <stdio.h>
#include <stdlib.h>
#include "cublasGEMM.h"

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// for timing
#include <sys/time.h>

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

int main(int argc, char** argv) {
    double *a, *b;
    float *c;
    int n;
    int a_fd, b_fd;
    struct timeval h_start, h_end;
    long duration;

    if (argc < 3) {
        printf("usage: %s <sequence format path> <matrix size>\n", argv[0]);
        exit(1);
    }

    // GEMM configuration.
    a_fd = open(argv[1], O_RDONLY);
    b_fd = open(argv[1], O_RDONLY);

    n = atoi(argv[2]);

    a = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, a_fd, 0);
    b = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, b_fd, 0);
    c = (float *) calloc(n * n, sizeof(float));

    printf("calculating the answer...\n");
    gettimeofday(&h_start, NULL);
    wholeMatrix_Dgemm(n, n, n, a, b, c);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("sequential format GEMM duration: %f ms\n", (float) duration / 1000);

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
    cpu_verify(a, b, c, n, n, n);

    munmap(a, sizeof(double) * n * n);
    munmap(b, sizeof(double) * n * n);
    close(a_fd);
    close(b_fd);
    free(c);
    return 0;
}