#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>

#include <sys/mman.h>

void generate_data(double *array, uint64_t size) {
    uint64_t i;
    for (i = 0; i < size; i++) {
        array[i] = (((double) rand())/RAND_MAX - 0.5)*100;
    }
}

int main(int argc, char** argv) {
    int64_t n;
    int seq_fd;
    double *B;
    int ret;
    size_t dsize = 0;
    if (argc < 3) {
        printf("usage: %s <seq format output path> <matrix size>\n", argv[0]);
        return 1;
    }

    srand(5);

    seq_fd = open(argv[1], O_RDWR | O_CREAT | O_TRUNC, 0644);
    n = atoll(argv[2]);

    dsize = sizeof(double) * n * n * n;

    ret = posix_fallocate(seq_fd, 0, dsize);
    if (ret != 0) {
        printf("fallocate error code: %d\n", ret);
        exit(ret);
    }

    // MAP_SHARED committed back to the original file but MAP_PRIVATE only writes to memory.
    // ref: https://stackoverflow.com/questions/9519648/what-is-the-difference-between-map-shared-and-map-private-in-the-mmap-function
    B = (double *) mmap(NULL, dsize, PROT_READ | PROT_WRITE, MAP_SHARED, seq_fd, 0);
    generate_data(B, n * n * n);

    msync(B, dsize, MS_SYNC);
    munmap(B, dsize);
    close(seq_fd);    
    return 0;
}