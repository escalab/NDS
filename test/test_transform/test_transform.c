#include "tensorstore.h"

int main(void) {
    int m = 8;
    int sub_m = 2;
    int i, j;
    double *seq_matrix, *tensor_matrix, *reformat_matrix;
    seq_matrix = (double *) malloc(sizeof(double) * m * m);
    tensor_matrix = (double *) calloc(m * m, sizeof(double));
    reformat_matrix = (double *) calloc(m * m, sizeof(double));

    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            seq_matrix[i * m + j] = i * m + j;
        }
    }

    seq2tensor(seq_matrix, tensor_matrix, m, m, sub_m, sub_m);

    printf("result of seq_matrix\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            printf("%.0f ", seq_matrix[i * m + j]);
        }
        printf("\n");
    }

    printf("result of tensor_matrix\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            printf("%.0f ", tensor_matrix[i * m + j]);
        }
        printf("\n");
    }

    tensor2seq(tensor_matrix, reformat_matrix, m, m, sub_m, sub_m);
    printf("result of tensor_matrix\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            printf("%.0f ", reformat_matrix[i * m + j]);
        }
        printf("\n");
    }
    free(seq_matrix);
    free(tensor_matrix);
    free(reformat_matrix);
    return 0;
}