#include "tensorstore.h"

void tensor_matrix_transpose(double *tensor_matrix, double *out_matrix, size_t n_row, size_t n_col, size_t sub_m, size_t sub_n) {
    size_t i, j;
    size_t cross_row = sub_m * n_col, cross_col = sub_m * sub_n;
    double *upper_ptr, *lower_ptr, *out_upper_ptr, *out_lower_ptr;

    // iterate submatrix at upper triangles
    // for example 4 * 4 matrix: 0, 1, 2, 3 
    for (i = 0; i < n_row / sub_m; i++) {
        // (1, 2, 3), (2, 3), (3)
        for (j = i; j < n_col / sub_n; j++) {
            // 0, 1 -> 1
            upper_ptr = tensor_matrix + i * cross_row + j * cross_col;
            // 1, 0 -> 4
            lower_ptr = tensor_matrix + j * cross_row + i * cross_col; 

            out_upper_ptr = out_matrix + i * cross_row + j * cross_col;
            out_lower_ptr = out_matrix + j * cross_row + i * cross_col; 

            memcpy(out_upper_ptr, lower_ptr, sizeof(double) * sub_m * sub_n);
            memcpy(out_lower_ptr, upper_ptr, sizeof(double) * sub_m * sub_n);
        }
    }
}

void seq_matrix_transpose(double *seq_matrix, double *out_matrix, size_t n_row, size_t n_col, size_t sub_m, size_t sub_n) {
    size_t i, j, ii;
    double *upper_ptr, *lower_ptr, *out_upper_ptr, *out_lower_ptr;

    // iterate submatrix at upper triangles 
    for (i = 0; i < n_row / sub_m; i++) {
        for (j = i; j < n_col / sub_n; j++) {
            upper_ptr = seq_matrix + i * sub_m * n_col + j * sub_n;
            lower_ptr = seq_matrix + j * sub_m * n_col + i * sub_n; 

            out_upper_ptr = out_matrix + i * sub_m * n_col + j * sub_n;
            out_lower_ptr = out_matrix + j * sub_m * n_col + i * sub_n; 

            // iterate rows in the submatrix
            for (ii = 0; ii < sub_m; ii++) {
                memcpy(out_upper_ptr + ii * n_col, lower_ptr + ii * n_col, sizeof(double) * sub_n);
                memcpy(out_lower_ptr + ii * n_col, upper_ptr + ii * n_col, sizeof(double) * sub_n);
            }
        }
    }
}

void seq2tensor(double *seq_matrix, double *tensor_matrix, size_t m, size_t n, size_t sub_m, size_t sub_n) {
    size_t i, j, ii;
    size_t cross_row = m, cross_col = sub_m;
    double *src, *dst;
    for (i = 0; i < m; i += sub_m) {
        for (j = 0; j < n; j += sub_n) {
            dst = tensor_matrix + i * cross_row + j * cross_col;
            src = seq_matrix + i * n + j;
            for (ii = 0; ii < sub_m; ii++) {
                memcpy((dst + ii * sub_n), (src + ii * n), sizeof(double) * sub_n);
            }
        }
    }  
}

void tensor2seq(double *tensor_matrix, double *seq_matrix, size_t m, size_t n, size_t sub_m, size_t sub_n) {
    size_t i, j, ii, idx = 0;
    for (i = 0; i < n; i += sub_m) {
        for (j = 0; j < n; j += sub_n) {
            for(ii = i; ii < i + sub_m; ii++) {
                memcpy((seq_matrix + ii * n + j), (tensor_matrix + idx), sizeof(double) * sub_n);
                idx += sub_n;
            }
        }
    }
}