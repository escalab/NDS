#include "tensorstore.h"
#include <stdint.h>
#include <sys/time.h>

void request_submatrix(double *seq_matrix, double *tensor_matrix, uint64_t y, uint64_t m, uint64_t sub_m) {
    uint64_t i, offset;

    offset = y * m * sub_m;
    for (i = 0; i < sub_m; i++) {
        memcpy(tensor_matrix + i * m, seq_matrix + offset, sizeof(double) * m);
        offset += m;
    }
}

int main(int argc, char** argv) {
    uint64_t m, sub_m, i;
    double *seq_matrix, *tensor_matrix;
    
    // time
    struct timeval start, end;
    uint64_t duration = 0;

    if (argc < 3) {
        printf("usage: %s <m> <sub_m>\n", argv[0]);
        exit(1);
    }
    
    m = (uint64_t) atoll(argv[1]);
    sub_m = (uint64_t) atoll(argv[2]);

    seq_matrix = (double *) malloc(sizeof(double) * m * m);
    tensor_matrix = (double *) calloc(m * sub_m, sizeof(double));

    memset(seq_matrix, 128, sizeof(double) * m * m);

    gettimeofday(&start, NULL);
    for (i = 0; i < m / sub_m; i++) {
        // printf("i: %lu\n", i);
        request_submatrix(seq_matrix, tensor_matrix, i, m, sub_m);
    }
    gettimeofday(&end, NULL);
    duration = ((end.tv_sec*1000000 + end.tv_usec) - (start.tv_sec*1000000 + start.tv_usec));
    printf("Reshaping time: %f (msec)\n", (double) duration / 1000.);

    free(seq_matrix);
    free(tensor_matrix);
    return 0;
}