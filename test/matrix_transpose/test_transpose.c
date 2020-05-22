#include "tensorstore.h"

// file I/O
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// timing
#include <sys/time.h>

int main(int argc, char** argv) {
    size_t m, sub_m;
    double *seq_matrix, *tensor_matrix;
    int seq_fd, tensor_fd;
    double *seq_file, *tensor_file;
    struct timeval h_start, h_end;
    unsigned long long duration;

    if (argc < 5) {
        printf("usage: %s <matrix size> <submatrix size> <seq_matrix_path> <tensor_matrix_path>\n", argv[0]);
        exit(1);
    }

    m = atoi(argv[1]);
    sub_m = atoi(argv[2]);

    if (m < sub_m) {
        printf("matrix size has to be larger than submatrix size\n");
        exit(1);
    }

    if (m % sub_m) {
        printf("submatrix size cannot divide matrix size evenly\n");
        exit(1);
    }

    seq_fd = open(argv[3], O_RDONLY, 0644);
    tensor_fd = open(argv[4], O_RDONLY, 0644);
    seq_file = (double *) mmap(NULL, sizeof(double) * m * m, PROT_READ, MAP_PRIVATE, seq_fd, 0);
    tensor_file = (double *) mmap(NULL, sizeof(double) * m * m, PROT_READ, MAP_PRIVATE, tensor_fd, 0);
    close(seq_fd);
    close(tensor_fd);

    seq_matrix = (double *) malloc(sizeof(double) * m * m);
    tensor_matrix = (double *) calloc(m * m, sizeof(double));

    gettimeofday(&h_start, NULL);
    tensor_matrix_transpose(tensor_file, tensor_matrix, m, m, sub_m, sub_m);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("tensor_matrix transpose time: %f ms\n", (float) duration / 1000);

#ifdef DEBUG
    printf("result of transposed tensor_matrix\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            printf("%f ", tensor_matrix[i * m + j]);
        }
        printf("\n");
    }
#endif

    gettimeofday(&h_start, NULL);
    seq_matrix_transpose(seq_file, seq_matrix, m, m, sub_m, sub_m);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   
    printf("seq_matrix transpose time: %f ms\n", (float) duration / 1000);

#ifdef DEBUG
    size_t i, j;
    printf("result of seq_matrix\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            printf("%f ", seq_matrix[i * m + j]);
        }
        printf("\n");
    }
#endif

    munmap(seq_file, sizeof(double) * m * m);
    munmap(tensor_file, sizeof(double) * m * m);
    free(seq_matrix);
    free(tensor_matrix);
    return 0;
}