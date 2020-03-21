#ifndef TENSORSTORE_H_
#define TENSORSTORE_H_
#include <stdio.h>
#include <stdlib.h>
void extract_submatrix(double *seq_matrix, double *sub_matrix, long long col, long long sub_m, long long sub_n);

void seq2tensor(double *seq_matrix, double *tensor_matrix, long long m, long long n, long long sub_m, long long sub_n);
void tensor2seq(double *tensor_matrix, double *seq_matrix, long long m, long long n, long long sub_m, long long sub_n);
#endif