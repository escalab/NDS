#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>

#include <sys/mman.h>

void float_tensor2seq(float *tensor_matrix, float *seq_matrix, size_t m, size_t n, size_t sub_m, size_t sub_n) {
    size_t i, j, ii, idx = 0;
    for (i = 0; i < n; i += sub_m) {
        for (j = 0; j < n; j += sub_n) {
            for(ii = i; ii < i + sub_m; ii++) {
                memcpy((seq_matrix + ii * n + j), (tensor_matrix + idx), sizeof(float) * sub_n);
                idx += sub_n;
            }
        }
    }
}

int main(int argc, char** argv) {
    int64_t n, sub_n;
    int seq_fd, tensor_fd;
    float *seq_matrix;
    float *tensor_matrix;
    int ret;
    size_t dsize = 0;
    if (argc < 5) {
        printf("usage: %s <tensor path> <seq path> <matrix size> <tensor_matrix size>\n", argv[0]);
        return 1;
    }

    srand(5);

    tensor_fd = open(argv[1], O_RDWR);
    seq_fd = open(argv[2], O_RDWR | O_CREAT | O_TRUNC, 0644);
    n = atoll(argv[3]);
    sub_n = atoll(argv[4]);

    dsize = sizeof(float) * n * n;

    ret = posix_fallocate(tensor_fd, 0, dsize);
    if (ret != 0) {
        printf("fallocate error code: %d\n", ret);
        exit(ret);
    }

    // MAP_SHARED committed back to the original file but MAP_PRIVATE only writes to memory.
    // ref: https://stackoverflow.com/questions/9519648/what-is-the-difference-between-map-shared-and-map-private-in-the-mmap-function
    seq_matrix = (float *) mmap(NULL, dsize, PROT_READ, MAP_PRIVATE, seq_fd, 0);
    tensor_matrix = (float *) mmap(NULL, dsize, PROT_READ | PROT_WRITE, MAP_SHARED, tensor_fd, 0);

    printf("assigning tensor_matrix\n");
    float_tensor2seq(tensor_matrix, seq_matrix, n, n, sub_n, sub_n);

    msync(tensor_matrix, dsize, MS_SYNC);
    munmap(seq_matrix, dsize);
    munmap(tensor_matrix, dsize);
    close(seq_fd);
    close(tensor_fd);
    return 0;
}