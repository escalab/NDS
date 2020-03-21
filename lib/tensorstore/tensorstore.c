#include "tensorstore.h"

void extract_submatrix(double *seq_matrix, double *sub_matrix, long long col, long long sub_m, long long sub_n) {
    long long ii, jj, idx;
    idx = 0;     
    for(ii = 0; ii < sub_m; ii++) {
        for(jj = 0; jj < sub_n; jj++) {
            sub_matrix[idx] = seq_matrix[ii * col + jj];
            idx++;
        }
    }
}

void seq2tensor(double *seq_matrix, double *tensor_matrix, long long m, long long n, long long sub_m, long long sub_n) {
    long long i, j;
    long long cross_row = m, cross_col = sub_m;
    for(i = 0; i < m; i+= sub_m) {
        for(j = 0; j < n; j+= sub_n) {
            extract_submatrix((seq_matrix + i * n + j), (tensor_matrix + i * cross_row + j * cross_col), n, sub_m, sub_n);
        }
    }  
}

void tensor2seq(double *tensor_matrix, double *seq_matrix, long long m, long long n, long long sub_m, long long sub_n) {
    long long i, j, ii, jj, count = 0;
    for (i = 0; i < n; i += sub_n) {
        for (j = 0; j < n; j += sub_n) {  
            for(ii = i; ii < i + sub_n; ii++) {
                for(jj = j; jj < j + sub_n; jj++) {
                    seq_matrix[ii * n + jj] = tensor_matrix[count];
                    count++;
                }
            }
        }
    }
}