#include "tensorstore.h"

void extract_submatrix(double *seq_matrix, double *sub_matrix, int col, int sub_m, int sub_n) {
    int ii, jj, idx;
    idx = 0;     
    for(ii = 0; ii < sub_m; ii++) {
        for(jj = 0; jj < sub_n; jj++) {
            sub_matrix[idx] = seq_matrix[ii * col + jj];
            idx++;
        }
    }
}

void seq2tensor(double *seq_matrix, double *tensor_matrix, int m, int n, int sub_m, int sub_n) {
    int i, j, ii, jj, idx;
    int cross_row = m, cross_col = sub_m;
    for(i = 0; i < m; i+= sub_m) {
        for(j = 0; j < n; j+= sub_n) {
            extract_submatrix((seq_matrix + i * n + j), (tensor_matrix + i * cross_row + j * cross_col), n, sub_m, sub_n);
        }
    }  
}

int main(void) {
    int m = 8;
    int sub_m = 2;
    int i, j;
    double *seq_matrix, *tensor_matrix;
    seq_matrix = (double *) malloc(sizeof(double) * m * m);
    tensor_matrix = (double *) calloc(m * m, sizeof(double));

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
    free(seq_matrix);
    free(tensor_matrix);
    return 0;
}