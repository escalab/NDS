#include "tensorstore.h"

int main(int argc, char** argv) {
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

    printf("result of seq_matrix\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            printf("%.0f ", seq_matrix[i * m + j]);
        }
        printf("\n");
    }

    seq2tensor(seq_matrix, tensor_matrix, m, m, sub_m, sub_m);

    printf("result of tensor_matrix\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            printf("%.0f ", tensor_matrix[i * m + j]);
        }
        printf("\n");
    }

    seq_matrix_transpose(seq_matrix, m, m, sub_m, sub_m);

    printf("result of transposed seq_matrix\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            printf("%.0f ", seq_matrix[i * m + j]);
        }
        printf("\n");
    }

    tensor_matrix_transpose(tensor_matrix, m, m, sub_m, sub_m);

    printf("result of transposed tensor_matrix\n");
    for (i = 0; i < m; i++) {
        for (j = 0; j < m; j++) {
            printf("%.0f ", tensor_matrix[i * m + j]);
        }
        printf("\n");
    }
    // tensor2seq(tensor_matrix, reformat_matrix, m, m, sub_m, sub_m);
    // printf("result of tensor_matrix\n");
    // for (i = 0; i < m; i++) {
    //     for (j = 0; j < m; j++) {
    //         printf("%.0f ", reformat_matrix[i * m + j]);
    //     }
    //     printf("\n");
    // }
    free(seq_matrix);
    free(tensor_matrix);
    free(reformat_matrix);
    return 0;
}