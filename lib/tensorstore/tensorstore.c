#include "tensorstore.h"

void extract_submatrix_from_seq(double *seq_matrix, double *sub_matrix, size_t n_col, size_t sub_m, size_t sub_n) {
    size_t ii, jj, idx;
    idx = 0;     
    for(ii = 0; ii < sub_m; ii++) {
        for(jj = 0; jj < sub_n; jj++) {
            sub_matrix[idx] = seq_matrix[ii * n_col + jj];
            idx++;
        }
    }
}

void seq_matrix_transpose(double *seq_matrix, size_t n_row, size_t n_col, size_t sub_m, size_t sub_n) {
    size_t i, j, ii;
    double temp[sub_n];
    double *upper_ptr, *lower_ptr;
    // iterate submatrix at upper triangles 
    for (i = 0; i < n_row / sub_m; i++) {
        for (j = i+1; j < n_col / sub_n; j++) {
            upper_ptr = seq_matrix + i * sub_m * n_col + j * sub_n;
            lower_ptr = seq_matrix + j * sub_m * n_col + i * sub_n; 
            // iterate rows in the submatrix
            for (ii = 0; ii < sub_m; ii++) {
                // swap elements in a row
                memcpy(&temp, upper_ptr + ii * n_col, sizeof(double) * sub_n);
                memcpy(upper_ptr + ii * n_col, lower_ptr + ii * n_col, sizeof(double) * sub_n);
                memcpy(lower_ptr + ii * n_col, &temp, sizeof(double) * sub_n);
            }
        }
    }
}

void seq2tensor(double *seq_matrix, double *tensor_matrix, size_t m, size_t n, size_t sub_m, size_t sub_n) {
    size_t i, j;
    size_t cross_row = m, cross_col = sub_m;
    for(i = 0; i < m; i+= sub_m) {
        for(j = 0; j < n; j+= sub_n) {
            extract_submatrix_from_seq((seq_matrix + i * n + j), (tensor_matrix + i * cross_row + j * cross_col), n, sub_m, sub_n);
        }
    }  
}

void tensor2seq(double *tensor_matrix, double *seq_matrix, size_t m, size_t n, size_t sub_m, size_t sub_n) {
    size_t i, j, ii, jj, count = 0;
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