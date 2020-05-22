#ifndef TENSORSTORE_H_
#define TENSORSTORE_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void extract_submatrix_from_seq(double *seq_matrix, double *sub_matrix, size_t col, size_t sub_m, size_t sub_n);
void seq_matrix_transpose(double *seq_matrix, size_t n_row, size_t n_col, size_t sub_m, size_t sub_n);
void tensor_matrix_transpose(double *tensor_matrix, size_t n_row, size_t n_col, size_t sub_m, size_t sub_n);
void seq2tensor(double *seq_matrix, double *tensor_matrix, size_t m, size_t n, size_t sub_m, size_t sub_n);
void tensor2seq(double *tensor_matrix, double *seq_matrix, size_t m, size_t n, size_t sub_m, size_t sub_n);
#endif