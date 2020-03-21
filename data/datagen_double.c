#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#include <sys/mman.h>

#include "tensorstore.h"

int main(int argc, char** argv) {
    long long i, n, sub_n;
    int seq_fd, tensor_fd;
    double *seq_matrix;
    double *tensor_matrix;
    int ret;
    size_t dsize = 0;
    if (argc < 5) {
        printf("usage: %s <normal output path> <block output path> <matrix size> <tensor_matrix size>\n", argv[0]);
        return 1;
    }

    srand(5);

    seq_fd = open(argv[1], O_RDWR | O_CREAT | O_TRUNC, 0644);
    tensor_fd = open(argv[2], O_RDWR | O_CREAT | O_TRUNC, 0644);
    n = atoll(argv[3]);
    sub_n = atoll(argv[4]);

    dsize = sizeof(double) * n * n;

    ret = posix_fallocate(seq_fd, 0, dsize);
    if (ret != 0) {
        printf("fallocate error code: %d\n", ret);
        exit(ret);
    }

    ret = posix_fallocate(tensor_fd, 0, dsize);
    if (ret != 0) {
        printf("fallocate error code: %d\n", ret);
        exit(ret);
    }

    // MAP_SHARED committed back to the original file but MAP_PRIVATE only writes to memory.
    // ref: https://stackoverflow.com/questions/9519648/what-is-the-difference-between-map-shared-and-map-private-in-the-mmap-function
    seq_matrix = (double *) mmap(NULL, dsize, PROT_READ | PROT_WRITE, MAP_SHARED, seq_fd, 0);
    tensor_matrix = (double *) mmap(NULL, dsize, PROT_READ | PROT_WRITE, MAP_SHARED, tensor_fd, 0);
    
    printf("assigning seq_matrix\n");
    for (i = 0; i < n * n; i++) {
        seq_matrix[i] = (double) rand() / RAND_MAX;
    }
    msync(seq_matrix, dsize, MS_SYNC);

    printf("assigning tensor_matrix\n");
    seq2tensor(seq_matrix, tensor_matrix, n, n, sub_n, sub_n);

    msync(tensor_matrix, dsize, MS_SYNC);
    munmap(seq_matrix, dsize);
    munmap(tensor_matrix, dsize);
    close(seq_fd);
    close(tensor_fd);
    return 0;
}