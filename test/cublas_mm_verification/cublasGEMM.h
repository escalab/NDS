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

void tensor_blockGemmEx(int x, int y, int z, int sub_m, int sub_n, int sub_k, double *a, double *b, float *c, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t computetype);
void tensor_blockSgemm_half(int x, int y, int z, int sub_m, int sub_n, int sub_k, double *a, double *b, float *c);
void tensor_blockSgemm(int x, int y, int z, int sub_m, int sub_n, int sub_k, double *a, double *b, float *c);
void tensor_blockDgemm(int x, int y, int z, int sub_m, int sub_n, int sub_k, double *a, double *b, float *c);

void sequential_blockGemmEx(int x, int y, int z, int sub_m, int sub_n, int sub_k, double *a, double *b, float *c, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t computetype);
void sequential_blockSgemm_half(int x, int y, int z, int sub_m, int sub_n, int sub_k, double *a, double *b, float *c);
void sequential_blockSgemm(int x, int y, int z, int sub_m, int sub_n, int sub_k, double *a, double *b, float *c);
void sequential_blockDgemm(int x, int y, int z, int sub_m, int sub_n, int sub_k, double *a, double *b, float *c);

void wholeMatrix_GemmEx(int m, int n, int k, const double *a, const double *b, float *c, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t computetype);
void wholeMatrix_Sgemm_half(int m, int n, int k, const double *a, const double *b, float *c);
void wholeMatrix_Sgemm(int m, int n, int k, const double *a, const double *b, float *c);
void wholeMatrix_Dgemm(int m, int n, int k, const double *a, const double *b, float *c);

#endif