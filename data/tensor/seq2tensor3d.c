#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>

#include <sys/mman.h>

void seq2tensor(const double *seq_matrix, double *tensor_matrix, size_t n, size_t sub_n) {
    size_t i, j, ii, jj, count = 0;
    for (i = 0; i < n * n; i += sub_n) {
        for (j = 0; j < n; j += sub_n) {
            for (ii = i; ii < i+sub_n; ii++) {
                for (jj = j; jj < j+sub_n; jj++) {
                    tensor_matrix[count] = seq_matrix[ii * n + jj];
                    // printf("%f ", tensor_matrix[count]);
                    count++;
                }
            }
            // printf("\n");
        }
    }  
}

int main(int argc, char** argv) {
    int64_t n, sub_n;
    int seq_fd, tensor_fd;
    double *seq_matrix;
    double *tensor_matrix;
    int ret;
    size_t dsize = 0;
    if (argc < 5) {
        printf("usage: %s <seq path> <tensor path> <tensor size> <building block size>\n", argv[0]);
        return 1;
    }

    srand(5);

    seq_fd = open(argv[1], O_RDWR);
    tensor_fd = open(argv[2], O_RDWR | O_CREAT | O_TRUNC, 0644);
    n = atoll(argv[3]);
    sub_n = atoll(argv[4]);

    dsize = sizeof(double) * n * n * n;

    ret = posix_fallocate(tensor_fd, 0, dsize);
    if (ret != 0) {
        printf("fallocate error code: %d\n", ret);
        exit(ret);
    }

    // MAP_SHARED committed back to the original file but MAP_PRIVATE only writes to memory.
    // ref: https://stackoverflow.com/questions/9519648/what-is-the-difference-between-map-shared-and-map-private-in-the-mmap-function
    seq_matrix = (double *) mmap(NULL, dsize, PROT_READ, MAP_PRIVATE, seq_fd, 0);
    tensor_matrix = (double *) mmap(NULL, dsize, PROT_READ | PROT_WRITE, MAP_SHARED, tensor_fd, 0);

    printf("\nassigning tensor_matrix\n");
    seq2tensor(seq_matrix, tensor_matrix, n, sub_n);

    msync(tensor_matrix, dsize, MS_SYNC);
    munmap(seq_matrix, dsize);
    munmap(tensor_matrix, dsize);
    close(seq_fd);
    close(tensor_fd);
    return 0;
}