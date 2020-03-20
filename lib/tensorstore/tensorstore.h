#ifndef TENSORSTORE_H_
#define TENSORSTORE_H_
#include <stdio.h>
#include <stdlib.h>
void extract_submatrix(double *seq_matrix, double *sub_matrix, int col, int sub_m, int sub_n);

void seq2tensor(double *seq_matrix, double *tensor_matrix, int m, int n, int sub_m, int sub_n);
void tensor2seq(double *tensor_matrix, double *seq_matrix, int m, int n, int sub_m, int sub_n);
#endif