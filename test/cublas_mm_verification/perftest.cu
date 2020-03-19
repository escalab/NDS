#include <stdio.h>
#include <stdlib.h>
#include "cublasGEMM.h"

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// for timing
#include <sys/time.h>

#ifndef ITER_NUM
#define ITER_NUM 10
#endif

int main(int argc, char** argv) {
    double *a, *b;
    float *c;
    int n, sub_n;
    int a_fd, b_fd;
    struct timeval h_start, h_end;
    long duration;

    if (argc < 3) {
        printf("usage: %s <matrix path> <matrix size> [submatrix size]\n", argv[0]);
        exit(1);
    }

    // GEMM configuration.
    a_fd = open(argv[1], O_RDONLY);
    b_fd = open(argv[1], O_RDONLY);
    n = atoi(argv[2]);

    // if users put size of submatrix, that means they are using tensor format.
    if (argc > 3) {
        sub_n = atoi(argv[3]);
    }

    a = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, a_fd, 0);
    b = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, b_fd, 0);
    c = (float *) calloc(n * n, sizeof(float));
    
    printf("calculating the answer...\n");
    for (int i = 0; i < ITER_NUM; i++) {
        memset(c, 0, n * n * sizeof(float));
        gettimeofday(&h_start, NULL);
        wholeMatrix_Sgemm_half(n, n, n, a, b, c);
        // sequential_blockSgemm_half(n, n, n, sub_n, sub_n, sub_n, a, b, c);
        // tensor_blockSgemm_half(n, n, n, sub_n, sub_n, sub_n, a_tensor, b_tensor, c);
        gettimeofday(&h_end, NULL);
        duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
        printf("sequential format GEMM duration: %f ms\n", (float) duration / 1000);    
    }

    munmap(a, sizeof(double) * n * n);
    munmap(b, sizeof(double) * n * n);
    close(a_fd);
    close(b_fd);
    free(c);
    return 0;
}