#include "cublasGEMM.h"

__global__ void d2f_kernel(double *din, float *dout, int dsize) {
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
	if (idx < dsize)
	{
		dout[idx] = din[idx];
	}
}

__global__ void d2h_kernel(double *din, half *dout, int dsize) {
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
	if (idx < dsize)
	{
		dout[idx] = din[idx];
	}
}

__global__ void h2f_kernel(half *din, float *dout, int dsize) {
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
	if (idx < dsize)
	{
		dout[idx] = din[idx];
	}
}

float* tensor_blockGemmEx(int x, int y, int z, int sub_m, int sub_n, int sub_k, 
    double *a, double *b, float *c) {
    int i, j, k;
    int cross_row = x * sub_k, cross_col = sub_m * sub_k;
    float alpha = 1.0;
    float beta = 1.0;
    float *a_sub_f, *b_sub_f, *c_sub_f;
    double *a_sub_d, *b_sub_d;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    cudaMalloc((void **) &a_sub_d, sizeof(double) * sub_m * sub_k);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * sub_k * sub_n);
    cudaMalloc((void **) &a_sub_f, sizeof(float) * sub_m * sub_k);
    cudaMalloc((void **) &b_sub_f, sizeof(float) * sub_k * sub_n);
    cudaMalloc((void **) &c_sub_f, sizeof(float) * sub_m * sub_n);

    int dsize = sub_m * sub_n;

    // custom block gemm
    for (i = 0; i < (x / sub_m); i++) {
        for (j = 0; j < (y / sub_n); j++) {
            cudaMemset(c_sub_f, 0, sub_m * sub_n * sizeof(float));
            for (k = 0; k < (z / sub_k); k++) {
                // here we can use GPUDirect?
                cudaMemcpy(a_sub_d, (a + i * cross_row + k * cross_col), sub_m * sub_k * sizeof(double), cudaMemcpyHostToDevice);    
                cudaMemcpy(b_sub_d, (b + k * cross_row + j * cross_col), sub_k * sub_n * sizeof(double), cudaMemcpyHostToDevice);
                d2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(a_sub_d, a_sub_f, dsize);
                d2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(b_sub_d, b_sub_f, dsize);
                // async execution (ref: https://forums.developer.nvidia.com/t/async-cublas/2837)
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, sub_k, &alpha, b_sub_f, CUDA_R_16F, sub_k, a_sub_f, CUDA_R_16F, sub_m, &beta, c_sub_f, CUDA_R_32F, sub_m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }
            cudaMemcpy((c + i * cross_row + j * cross_col), c_sub_f, sub_m * sub_n * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }

    cublasDestroy(handle);

    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(a_sub_f);
    cudaFree(b_sub_f);
    cudaFree(c_sub_f);
    return c;
}

// DON'T USE. Lose precision somewhere.
float* tensor_blockHgemm(int x, int y, int z, int sub_m, int sub_n, int sub_k, 
    double *a, double *b, float *c) {
    int i, j, k;
    int cross_row = x * sub_k, cross_col = sub_m * sub_k;
    half alpha = 1.0;
    half beta = 1.0;
    half *a_sub_h, *b_sub_h, *c_sub_h;
    double *a_sub_d, *b_sub_d;
    float *c_sub_d;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMalloc((void **) &a_sub_d, sizeof(double) * sub_m * sub_k);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * sub_k * sub_n);
    cudaMalloc((void **) &c_sub_d, sizeof(float) * sub_m * sub_n);
    cudaMalloc((void **) &a_sub_h, sizeof(half) * sub_m * sub_k);
    cudaMalloc((void **) &b_sub_h, sizeof(half) * sub_k * sub_n);
    cudaMalloc((void **) &c_sub_h, sizeof(half) * sub_m * sub_n);

    int dsize = sub_m * sub_n;

    // custom block gemm
    for (i = 0; i < (x / sub_m); i++) {
        for (j = 0; j < (y / sub_n); j++) {
            cudaMemset(c_sub_h, 0, sub_m * sub_n * sizeof(half));
            for (k = 0; k < (z / sub_k); k++) {
                // here we can use GPUDirect?
                cudaMemcpy(a_sub_d, (a + i * cross_row + k * cross_col), sub_m * sub_k * sizeof(double), cudaMemcpyHostToDevice);    
                cudaMemcpy(b_sub_d, (b + k * cross_row + j * cross_col), sub_k * sub_n * sizeof(double), cudaMemcpyHostToDevice);
                d2h_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(a_sub_d, a_sub_h, dsize);
                d2h_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(b_sub_d, b_sub_h, dsize);
                cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, sub_k, &alpha, b_sub_h, sub_k, a_sub_h, sub_m, &beta, c_sub_h, sub_m);
            }
            h2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(c_sub_h, c_sub_d, dsize);
            cudaMemcpy((c + i * cross_row + j * cross_col), c_sub_d, sub_m * sub_n * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
    
    cublasDestroy(handle);

    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(c_sub_d);
    cudaFree(a_sub_h);
    cudaFree(b_sub_h);
    cudaFree(c_sub_h);

    return c;
}

float* tensor_blockSgemm(int x, int y, int z, int sub_m, int sub_n, int sub_k, 
    double *a, double *b, float *c) {
    int i, j, k;
    int cross_row = x * sub_k, cross_col = sub_m * sub_k;
    float alpha = 1.0;
    float beta = 1.0;
    float *a_sub_f, *b_sub_f, *c_sub_f;
    double *a_sub_d, *b_sub_d;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    cudaMalloc((void **) &a_sub_d, sizeof(double) * sub_m * sub_k);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * sub_k * sub_n);
    cudaMalloc((void **) &a_sub_f, sizeof(float) * sub_m * sub_k);
    cudaMalloc((void **) &b_sub_f, sizeof(float) * sub_k * sub_n);
    cudaMalloc((void **) &c_sub_f, sizeof(float) * sub_m * sub_n);

    int dsize = sub_m * sub_n;

    // custom block gemm
    for (i = 0; i < (x / sub_m); i++) {
        for (j = 0; j < (y / sub_n); j++) {
            cudaMemset(c_sub_f, 0, sub_m * sub_n * sizeof(float));
            for (k = 0; k < (z / sub_k); k++) {
                // here we can use GPUDirect?
                cudaMemcpy(a_sub_d, (a + i * cross_row + k * cross_col), sub_m * sub_k * sizeof(double), cudaMemcpyHostToDevice);    
                cudaMemcpy(b_sub_d, (b + k * cross_row + j * cross_col), sub_k * sub_n * sizeof(double), cudaMemcpyHostToDevice);
                d2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(a_sub_d, a_sub_f, dsize);
                d2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(b_sub_d, b_sub_f, dsize);
                // async execution (ref: https://forums.developer.nvidia.com/t/async-cublas/2837)
                cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, sub_k, &alpha, b_sub_f, CUDA_R_16F, sub_k, a_sub_f, CUDA_R_16F, sub_m, &beta, c_sub_f, CUDA_R_32F, sub_m);
            }
            cudaMemcpy((c + i * cross_row + j * cross_col), c_sub_f, sub_m * sub_n * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }

    cublasDestroy(handle);

    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(a_sub_f);
    cudaFree(b_sub_f);
    cudaFree(c_sub_f);
    return c;
}

float* tensor_blockDgemm(int x, int y, int z, int sub_m, int sub_n, int sub_k, 
    double *a, double *b, float *c) {
    int i, j, k;
    int cross_row = x * sub_k, cross_col = sub_m * sub_k;
    double alpha = 1.0;
    double beta = 1.0;
    double *a_sub_d, *b_sub_d, *c_sub_d;
    float *c_sub_f;
    cublasHandle_t handle;
    cublasCreate(&handle);


    cudaMalloc((void **) &a_sub_d, sizeof(double) * sub_m * sub_k);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * sub_k * sub_n);
    cudaMalloc((void **) &c_sub_d, sizeof(double) * sub_m * sub_n);
    cudaMalloc((void **) &c_sub_f, sizeof(float) * sub_m * sub_n);

    int dsize = sub_m * sub_n;

    // custom block gemm
    for (i = 0; i < (x / sub_m); i++) {
        for (j = 0; j < (y / sub_n); j++) {
            cudaMemset(c_sub_d, 0, sub_m * sub_n * sizeof(double));
            for (k = 0; k < (z / sub_k); k++) {
                // here we can use GPUDirect?
                cudaMemcpy(a_sub_d, (a + i * cross_row + k * cross_col), sub_m * sub_k * sizeof(double), cudaMemcpyHostToDevice);    
                cudaMemcpy(b_sub_d, (b + k * cross_row + j * cross_col), sub_k * sub_n * sizeof(double), cudaMemcpyHostToDevice);
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, sub_k, &alpha, b_sub_d, sub_k, a_sub_d, sub_m, &beta, c_sub_d, sub_m);
            }
            d2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(c_sub_d, c_sub_f, dsize);
            cudaMemcpy((c + i * cross_row + j * cross_col), c_sub_f, sub_m * sub_n * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }

    cublasDestroy(handle);

    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(c_sub_d);
    cudaFree(c_sub_f);

    return c;
}

float* sequential_blockSgemm(int x, int y, int z, int sub_m, int sub_n, int sub_k, 
    double *a, double *b, float *c) {
    int i, j, k, ii, kk, i_idx, k_idx;
    float alpha = 1.0;
    float beta = 1.0;
    double *a_sub_d, *b_sub_d;
    float *a_sub_f, *b_sub_f, *c_sub_f;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    cudaMalloc((void **) &a_sub_d, sizeof(double) * sub_m * sub_k);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * sub_k * sub_n);
    cudaMalloc((void **) &a_sub_f, sizeof(float) * sub_m * sub_k);
    cudaMalloc((void **) &b_sub_f, sizeof(float) * sub_k * sub_n);
    cudaMalloc((void **) &c_sub_f, sizeof(float) * sub_m * sub_n);

    int dsize = sub_m * sub_n;

    for (i = 0; i < x; i += sub_m) {
        for (j = 0; j < y; j += sub_n) {
            cudaMemset(c_sub_f, 0, sub_m * sub_n * sizeof(float));
            for (k = 0; k < z; k += sub_k) {
                for (ii = i, i_idx = 0; ii < (i + sub_m); ii++, i_idx++) {
                    cudaMemcpy((a_sub_d + i_idx * sub_n), (a + ii*y + k), sub_k * sizeof(double), cudaMemcpyHostToDevice);
                }

                for (kk = k, k_idx = 0; kk < (k + sub_k); kk++, k_idx++) {
                    cudaMemcpy((b_sub_d + k_idx * sub_n), (b + kk * y + j), sub_n * sizeof(double), cudaMemcpyHostToDevice);
                }

                d2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(a_sub_d, a_sub_f, dsize);
                d2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(b_sub_d, b_sub_f, dsize);
                // cublasDgemm EXPLANATION ------------------------------------------------
                // the memory layout is different from we know
                // a = [0 1; b = [3 2; 
                //      2 3]      1 0]
                // if use a_d then b_d, c[0][0] will be a[0, 0] * b[0, 0] + a[1, 0] * b[0, 1] = 4
                // with b_d then a_d, c[0][0] will be a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] = 1
                // maybe that's because inside GPU it uses column major storage.
                cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, sub_k, &alpha, b_sub_f, CUDA_R_16F, sub_k, a_sub_f, CUDA_R_16F, sub_m, &beta, c_sub_f, CUDA_R_32F, sub_m);
                // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, sub_k, &alpha, b_sub_f, sub_k, a_sub_f, sub_m, &beta, c_sub_f, sub_m);
            }
            for (ii = i, i_idx = 0; ii < (i + sub_n); ii++, i_idx++) {
                cudaMemcpy((c + ii * y + j), (c_sub_f + i_idx * sub_n), sub_n * sizeof(float), cudaMemcpyDeviceToHost);
            }                
        }
    }  
    
    cublasDestroy(handle);

    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(a_sub_f);
    cudaFree(b_sub_f);
    cudaFree(c_sub_f);

    return c;
}

float* sequential_blockDgemm(int x, int y, int z, int sub_m, int sub_n, int sub_k, 
    double *a, double *b, float *c) {
    int i, j, k, ii, kk, i_idx, k_idx;
    double alpha = 1.0;
    double beta = 1.0;
    double *a_sub_d, *b_sub_d, *c_sub_d;
    float *c_sub_f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMalloc((void **) &a_sub_d, sizeof(double) * sub_m * sub_k);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * sub_k * sub_n);
    cudaMalloc((void **) &c_sub_d, sizeof(double) * sub_m * sub_n);
    cudaMalloc((void **) &c_sub_f, sizeof(float) * sub_m * sub_n);

    int dsize = sub_m * sub_n;

    for (i = 0; i < x; i += sub_m) {
        for (j = 0; j < y; j += sub_n) {
            cudaMemset(c_sub_d, 0, sub_m * sub_n * sizeof(double));
            for (k = 0; k < z; k += sub_k) {
                for (ii = i, i_idx = 0; ii < (i + sub_m); ii++, i_idx++) {
                    cudaMemcpy((a_sub_d + i_idx * sub_n), (a + ii*y + k), sub_k * sizeof(double), cudaMemcpyHostToDevice);
                }

                for (kk = k, k_idx = 0; kk < (k + sub_k); kk++, k_idx++) {
                    cudaMemcpy((b_sub_d + k_idx * sub_n), (b + kk * y + j), sub_n * sizeof(double), cudaMemcpyHostToDevice);
                }
                // cublasDgemm EXPLANATION ------------------------------------------------
                // the memory layout is different from we know
                // a = [0 1; b = [3 2; 
                //      2 3]      1 0]
                // if use a_d then b_d, c[0][0] will be a[0, 0] * b[0, 0] + a[1, 0] * b[0, 1] = 4
                // with b_d then a_d, c[0][0] will be a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] = 1
                // maybe that's because inside GPU it uses column major storage.
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, sub_k, &alpha, b_sub_d, sub_k, a_sub_d, sub_m, &beta, c_sub_d, sub_m);
            }
            d2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(c_sub_d, c_sub_f, dsize);
            for (ii = i, i_idx = 0; ii < (i + sub_n); ii++, i_idx++) {
                cudaMemcpy((c + ii * y + j), (c_sub_f + i_idx * sub_n), sub_n * sizeof(float), cudaMemcpyDeviceToHost);
            }                
        }
    }  
    
    cublasDestroy(handle);

    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(c_sub_d);
    cudaFree(c_sub_f);

    return c;
}

// DON'T USE. Lose precision somewhere.
float* wholeMatrixHgemm(int m, int n, int k, const double *a, const double *b, float *c) {
    half alpha = 1.0;
    half beta = 0.0;
    double *a_d, *b_d;
    half *a_h, *b_h, *c_h;
    float *c_f;
    int dsize = m * n;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMalloc((void **) &a_d, sizeof(double) * m * k);
    cudaMalloc((void **) &b_d, sizeof(double) * k * n);
    cudaMalloc((void **) &a_h, sizeof(half) * m * k);
    cudaMalloc((void **) &b_h, sizeof(half) * k * n);
    cudaMalloc((void **) &c_h, sizeof(half) * k * n);
    cudaMalloc((void **) &c_f, sizeof(float) * m * n);

    cudaMemcpy(a_d, a, sizeof(double) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(double) * k * n, cudaMemcpyHostToDevice);

    d2h_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(a_d, a_h, dsize);
    d2h_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(b_d, b_h, dsize);

    cudaFree(a_d);
    cudaFree(b_d);
    // cublasDgemm EXPLANATION ------------------------------------------------
    // the memory layout is different from we know
    // a = [0 1; b = [3 2; 
    //      2 3]      1 0]
    // if use a_d then b_d, c[0][0] will be a[0, 0] * b[0, 0] + a[1, 0] * b[0, 1] = 4
    // with b_d then a_d, c[0][0] will be a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] = 1
    // maybe that's because inside GPU it uses column major storage.
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, b_h, k, a_h, m, &beta, c_h, m);
    cudaFree(a_h);
    cudaFree(b_h);
    h2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(c_h, c_f, dsize);
    cudaMemcpy(c, c_f, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    
    cublasDestroy(handle);
    cudaFree(c_h);
    cudaFree(c_f);
    return c;
}

float* wholeMatrixSgemm(int m, int n, int k, const double *a, const double *b, float *c) {
    float alpha = 1.0;
    float beta = 0.0;
    double *a_d, *b_d;
    float *a_f, *b_f, *c_f;
    int dsize = m * n;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    cudaMalloc((void **) &a_d, sizeof(double) * m * k);
    cudaMalloc((void **) &b_d, sizeof(double) * k * n);
    cudaMalloc((void **) &a_f, sizeof(float) * m * k);
    cudaMalloc((void **) &b_f, sizeof(float) * k * n);
    cudaMalloc((void **) &c_f, sizeof(float) * m * n);

    cudaMemcpy(a_d, a, sizeof(double) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(double) * k * n, cudaMemcpyHostToDevice);

    d2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(a_d, a_f, dsize);
    d2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(b_d, b_f, dsize);

    cudaFree(a_d);
    cudaFree(b_d);
    // cublasDgemm EXPLANATION ------------------------------------------------
    // the memory layout is different from we know
    // a = [0 1; b = [3 2; 
    //      2 3]      1 0]
    // if use a_d then b_d, c[0][0] will be a[0, 0] * b[0, 0] + a[1, 0] * b[0, 1] = 4
    // with b_d then a_d, c[0][0] will be a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] = 1
    // maybe that's because inside GPU it uses column major storage.
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, b_f, CUDA_R_16F, k, a_f, CUDA_R_16F, m, &beta, c_f, CUDA_R_32F, m);
    cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, b_f, CUDA_R_16F, k, a_f, CUDA_R_16F, m, &beta, c_f, CUDA_R_32F, m);

    cudaFree(a_f);
    cudaFree(b_f);

    cudaMemcpy(c, c_f, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    
    cublasDestroy(handle);

    cudaFree(c_f);
    return c;
}

float* wholeMatrixDgemm(int m, int n, int k, const double *a, const double *b, float *c) {
    double alpha = 1.0;
    double beta = 0.0;
    double *a_d, *b_d, *c_d;
    float *c_f;
    int dsize = m * n;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMalloc((void **) &a_d, sizeof(double) * m * k);
    cudaMalloc((void **) &b_d, sizeof(double) * k * n);
    cudaMalloc((void **) &c_d, sizeof(double) * m * n);

    cudaMemcpy(a_d, a, sizeof(double) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(double) * k * n, cudaMemcpyHostToDevice);

    // cublasDgemm EXPLANATION ------------------------------------------------
    // the memory layout is different from we know
    // a = [0 1; b = [3 2; 
    //      2 3]      1 0]
    // if use a_d then b_d, c[0][0] will be a[0, 0] * b[0, 0] + a[1, 0] * b[0, 1] = 4
    // with b_d then a_d, c[0][0] will be a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] = 1
    // maybe that's because inside GPU it uses column major storage.
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, b_d, k, a_d, m, &beta, c_d, m);
    cudaFree(a_d);
    cudaFree(b_d);

    cudaMalloc((void **) &c_f, sizeof(float) * m * n);
    d2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(c_d, c_f, dsize);
    cudaMemcpy(c, c_f, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    
    cublasDestroy(handle);

    cudaFree(c_d);
    cudaFree(c_f);
    return c;
}