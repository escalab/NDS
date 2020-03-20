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

void tensor2seq(double *tensor_matrix, double *seq_matrix, int m, int n, int sub_m, int sub_n) {
    int count = 0;
    for (int i = 0; i < n; i += sub_n) {
        for (int j = 0; j < n; j += sub_n) {  
            for(int ii = i; ii < i + sub_n; ii++) {
                for(int jj = j; jj < j + sub_n; jj++) {
                    seq_matrix[ii * n + jj] = tensor_matrix[count];
                    count++;
                }
            }
        }
    }
}