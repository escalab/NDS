#include <stdio.h>
#include <stdlib.h>
#include "cublasGEMM.h"

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// for timing
#include <sys/time.h>

// ALGO 0: wholeMatrix_Sgemm
// ALGO 1: wholeMatrix_Sgemm_half
// ALGO 2: sequential_blockSgemm
// ALGO 3: sequential_blockSgemm_half
// ALGO 4: tensor_blockSgemm
// ALGO 5: tensor_blockSgemm_half

#ifndef ALGO
#define ALGO 0
#endif

int main(int argc, char** argv) {
    double *a_f, *b_f, *a, *b;
    float *c;
    long long n;
    int a_fd, b_fd;
    struct timeval h_start, h_end;
    long duration;
    
#ifdef IS_BLOCK_ALGO
    long long sub_n;
    if (argc < 5) {
        printf("usage: %s <matrix A path> <matrix A path> <matrix size> <submatrix size>\n", argv[0]);
        exit(1);
    }
    sub_n = atoi(argv[4]);
#else
    if (argc < 4) {
        printf("usage: %s <matrix B path> <matrix B path> <matrix size>\n", argv[0]);
        exit(1);
    }
#endif

    // GEMM configuration.
    a_fd = open(argv[1], O_RDONLY);
    b_fd = open(argv[2], O_RDONLY);
    n = atoi(argv[3]);

    a_f = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, a_fd, 0);
    b_f = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, b_fd, 0);
    
    a = a_f;
    b = b_f;
    // cudaMallocHost((void**)&a, sizeof(double) * n * n);
    // cudaMallocHost((void**)&b, sizeof(double) * n * n);
    // memcpy(a, a_f, sizeof(double) * n * n);
    // memcpy(b, b_f, sizeof(double) * n * n);

    c = (float *) calloc(n * n, sizeof(float));
    printf("memory usage: input: %llu bytes, output: %llu bytes\n", sizeof(double) * n * n * 2, sizeof(float) * n * n);
    printf("calculating the answer...\n");
    printf("running algorithm %d\n", ALGO);
    gettimeofday(&h_start, NULL);
#ifndef IS_BLOCK_ALGO
#if ALGO == 0
    wholeMatrix_Sgemm(n, n, n, a, b, c);
#elif ALGO == 1
    wholeMatrix_Sgemm_half(n, n, n, a, b, c);
#endif
#else
#if ALGO == 2
    sequential_blockSgemm(n, n, n, sub_n, sub_n, sub_n, a, b, c);
#elif ALGO == 3
    sequential_blockSgemm_half(n, n, n, sub_n, sub_n, sub_n, a, b, c);
#elif ALGO == 4
    tensor_blockSgemm(n, n, n, sub_n, sub_n, sub_n, a, b, c);
#elif ALGO == 5
    tensor_blockSgemm_half(n, n, n, sub_n, sub_n, sub_n, a, b, c);
#elif ALGO == 6
    tensor_blockSgemm_half_async(n, n, n, sub_n, sub_n, sub_n, a, b, c);
#endif
#endif

    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("computation duration: %f ms\n", (float) duration / 1000);    
    // cudaFreeHost(a);
    // cudaFreeHost(b);
    munmap(a_f, sizeof(double) * n * n);
    munmap(b_f, sizeof(double) * n * n);
    close(a_fd);
    close(b_fd);
    free(c);
    return 0;
}