#ifndef CUBLASGEMM_H_
#define CUBLASGEMM_H_
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define THREADS_PER_BLOCK 256

__global__ void d2f_kernel(double *din, float *dout, int dsize);
__global__ void d2h_kernel(double *din, half *dout, int dsize);
__global__ void h2f_kernel(half *din, float *dout, int dsize);

float* tensor_blockGemmEx(int x, int y, int z, int sub_m, int sub_n, int sub_k, double *a, double *b, float *c);
float* tensor_blockSgemm(int x, int y, int z, int sub_m, int sub_n, int sub_k, double *a, double *b, float *c);
float* tensor_blockDgemm(int x, int y, int z, int sub_m, int sub_n, int sub_k, double *a, double *b, float *c);
float* sequential_blockSgemm(int x, int y, int z, int sub_m, int sub_n, int sub_k, double *a, double *b, float *c);
float* sequential_blockDgemm(int x, int y, int z, int sub_m, int sub_n, int sub_k, double *a, double *b, float *c);
float* wholeMatrixSgemm(int m, int n, int k, const double *a, const double *b, float *c);
float* wholeMatrixDgemm(int m, int n, int k, const double *a, const double *b, float *c);

#endif