#include "tensorstore.h"
// timing
#include <sys/time.h>

int main(int argc, char** argv) {
    size_t m, sub_m, i, j;
    double *seq_matrix, *tensor_matrix, *reformat_matrix;
    struct timeval h_start, h_end;
    unsigned long long duration;

    if (argc < 3) {
        printf("usage: %s <matrix size> <submatrix size>\n", argv[0]);
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

    seq_matrix = (double *) malloc(sizeof(double) * m * m);
    tensor_matrix = (double *) calloc(m * m, sizeof(double));
    reformat_matrix = (double *) calloc(m * m, sizeof(double));

    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            seq_matrix[i * m + j] = i * m + j;
        }
    }

#ifdef DEBUG
    printf("result of seq_matrix\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            printf("%.0f ", seq_matrix[i * m + j]);
        }
        printf("\n");
    }
#endif

    seq2tensor(seq_matrix, tensor_matrix, m, m, sub_m, sub_m);

#ifdef DEBUG
    printf("result of tensor_matrix\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            printf("%.0f ", tensor_matrix[i * m + j]);
        }
        printf("\n");
    }
#endif

    gettimeofday(&h_start, NULL);
    seq_matrix_transpose(seq_matrix, m, m, sub_m, sub_m);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   
    printf("seq_matrix transpose time: %f ms\n", (float) duration / 1000);

#ifdef DEBUG
    printf("result of transposed seq_matrix\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            printf("%.0f ", seq_matrix[i * m + j]);
        }
        printf("\n");
    }
#endif

    gettimeofday(&h_start, NULL);
    tensor_matrix_transpose(tensor_matrix, m, m, sub_m, sub_m);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("tensor_matrix transpose time: %f ms\n", (float) duration / 1000);

#ifdef DEBUG
    printf("result of transposed tensor_matrix\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            printf("%.0f ", tensor_matrix[i * m + j]);
        }
        printf("\n");
    }
#endif
   
    tensor2seq(tensor_matrix, reformat_matrix, m, m, sub_m, sub_m);
    
#ifdef DEBUG
    printf("result of reformat_matrix\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            printf("%.0f ", reformat_matrix[i * m + j]);
        }
        printf("\n");
    }
#endif
    free(seq_matrix);
    free(tensor_matrix);
    free(reformat_matrix);
    return 0;
}