#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>

#include <sys/mman.h>

int main(int argc, char** argv) {
    int64_t i, n;
    int seq_fd;
    float *seq_matrix;
    int ret;
    size_t dsize = 0;
    if (argc < 3) {
        printf("usage: %s <seq format output path> <matrix size>\n", argv[0]);
        return 1;
    }

    srand(5);

    seq_fd = open(argv[1], O_RDWR | O_CREAT | O_TRUNC, 0644);
    n = atoll(argv[2]);

    dsize = sizeof(float) * n * n;

    ret = posix_fallocate(seq_fd, 0, dsize);
    if (ret != 0) {
        printf("fallocate error code: %d\n", ret);
        exit(ret);
    }

    // MAP_SHARED committed back to the original file but MAP_PRIVATE only writes to memory.
    // ref: https://stackoverflow.com/questions/9519648/what-is-the-difference-between-map-shared-and-map-private-in-the-mmap-function
    seq_matrix = (float *) mmap(NULL, dsize, PROT_READ | PROT_WRITE, MAP_SHARED, seq_fd, 0);
    
    printf("assigning seq_matrix\n");
    for (i = 0; i < n * n; i++) {
        seq_matrix[i] = (float) rand() / RAND_MAX;
    }
    msync(seq_matrix, dsize, MS_SYNC);

    munmap(seq_matrix, dsize);
    close(seq_fd);
    return 0;
}
