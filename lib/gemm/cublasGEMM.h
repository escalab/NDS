#ifndef CUBLASGEMM_H_
#define CUBLASGEMM_H_
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

// timing
#include <sys/time.h>

#define THREADS_PER_BLOCK 256

__global__ void d2f_kernel(const double *din, float *dout, size_t dsize);
__global__ void d2h_kernel(const double *din, half *dout, size_t dsize);
__global__ void h2f_kernel(half *din, float *dout, size_t dsize);

void tensor_blockGemmEx_async(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, const double *a, const double *b, float *c, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t computetype);
void tensor_blockSgemm_half_async(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, const double *a, const double *b, float *c);

void tensor_blockGemmEx(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, const double *a, const double *b, float *c, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t computetype);
void tensor_blockSgemm_half(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, const double *a, const double *b, float *c);
void tensor_blockSgemm(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, const double *a, const double *b, float *c);
void tensor_blockDgemm(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, const double *a, const double *b, float *c);

void sequential_blockGemmEx(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, const double *a, const double *b, float *c, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t computetype);
void sequential_blockSgemm_half(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, const double *a, const double *b, float *c);
void sequential_blockSgemm(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, const double *a, const double *b, float *c);
void sequential_blockDgemm(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, const double *a, const double *b, float *c);

void wholeMatrix_GemmEx(size_t m, size_t n, size_t k, const double *a, const double *b, float *c, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t computetype);
void wholeMatrix_Sgemm_half(size_t m, size_t n, size_t k, const double *a, const double *b, float *c);
void wholeMatrix_Sgemm(size_t m, size_t n, size_t k, const double *a, const double *b, float *c);
void wholeMatrix_Dgemm(size_t m, size_t n, size_t k, const double *a, const double *b, float *c);

#endif