#include "cublasGEMM.h"

__global__ void d2f_kernel(const double *din, float *dout, size_t dsize) {
	size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
	if (idx < dsize)
	{
		dout[idx] = din[idx];
	}
}

__global__ void d2h_kernel(const double *din, half *dout, size_t dsize) {
	size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
	if (idx < dsize)
	{
		dout[idx] = din[idx];
	}
}

__global__ void h2f_kernel(half *din, float *dout, size_t dsize) {
	size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
	if (idx < dsize)
	{
		dout[idx] = din[idx];
	}
}

void tensor_blockGemmEx(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, 
    const double *a, const double *b, float *c, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t computetype) {
    size_t i, j, k;
    size_t cross_row = x * sub_k, cross_col = sub_m * sub_k;
    float alpha = 1.0;
    float beta = 1.0;
    double *a_sub_d, *b_sub_d;
    float *c_sub_f;
    struct timeval h_start, h_end;
    unsigned long long h2d_time = 0, d2h_time = 0, kernel_time = 0;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    // cudaStream_t stream[z / sub_k];
    // for (i = 0; i < z / sub_k; i++) {
    //     cudaStreamCreate(stream + i);
    // }
    
    cudaMalloc((void **) &a_sub_d, sizeof(double) * sub_m * sub_k);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * sub_k * sub_n);
    cudaMalloc((void **) &c_sub_f, sizeof(float) * sub_m * sub_n);

    // custom block gemm
    for (i = 0; i < (x / sub_m); i++) {
        for (j = 0; j < (y / sub_n); j++) {
            cudaMemset(c_sub_f, 0, sub_m * sub_n * sizeof(float));
            for (k = 0; k < (z / sub_k); k++) {
                // here we can use GPUDirect?
                gettimeofday(&h_start, NULL);
                cudaMemcpy(a_sub_d, (a + i * cross_row + k * cross_col), sub_m * sub_k * sizeof(double), cudaMemcpyHostToDevice);    
                cudaMemcpy(b_sub_d, (b + k * cross_row + j * cross_col), sub_k * sub_n * sizeof(double), cudaMemcpyHostToDevice);
                gettimeofday(&h_end, NULL);
                h2d_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);            
                // async execution (ref: https://forums.developer.nvidia.com/t/async-cublas/2837)
                // cudaDataType_t helps users to convert data inside the function call
                
                gettimeofday(&h_start, NULL);
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, sub_k, &alpha, b_sub_d, Btype, sub_k, a_sub_d, Atype, sub_m, &beta, c_sub_f, Ctype, sub_m, computetype, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                cudaDeviceSynchronize();
                gettimeofday(&h_end, NULL);
                kernel_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);            
            }
            gettimeofday(&h_start, NULL);
            cudaMemcpy((c + i * cross_row + j * cross_col), c_sub_f, sub_m * sub_n * sizeof(float), cudaMemcpyDeviceToHost);
            gettimeofday(&h_end, NULL);
            d2h_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);            
        }
    }

    cublasDestroy(handle);
    // for (i = 0; i < z / sub_k; i++) {
    //     cudaStreamDestroy(stream[i]);
    // }
    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(c_sub_f);
    printf("h2d time: %f ms\n", (float) h2d_time / 1000);
    printf("kernel time: %f ms\n", (float) kernel_time / 1000);
    printf("d2h time: %f ms\n", (float) d2h_time / 1000);
}

void tensor_blockSgemm_half(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, 
    const double *a, const double *b, float *c) {
    tensor_blockGemmEx(x, y, z, sub_m, sub_n, sub_k, a, b, c, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F);
}

void tensor_blockSgemm(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, 
    const double *a, const double *b, float *c) {
    tensor_blockGemmEx(x, y, z, sub_m, sub_n, sub_k, a, b, c, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F);
}

// DON'T USE. Lose precision somewhere.
float* tensor_blockHgemm(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, 
    const double *a, const double *b, float *c) {
    size_t i, j, k;
    size_t cross_row = x * sub_k, cross_col = sub_m * sub_k;
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

    size_t dsize = sub_m * sub_n;

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

void tensor_blockDgemm(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, 
    const double *a, const double *b, float *c) {    
    size_t i, j, k;
    size_t cross_row = x * sub_k, cross_col = sub_m * sub_k;
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

    size_t dsize = sub_m * sub_n;

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
}

void sequential_blockGemmEx(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, 
    const double *a, const double *b, float *c, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t computetype) {
    size_t i, j, k, ii, kk, i_idx, k_idx;
    float alpha = 1.0;
    float beta = 1.0;
    double *a_sub_d, *b_sub_d;
    float *c_sub_f;
    struct timeval h_start, h_end;
    unsigned long long h2d_time = 0, d2h_time = 0, kernel_time = 0;
    size_t in_pitch, out_pitch;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    // for copying data and running kernel asynchronously: https://devblogs.nvidia.com/how-overlap-data-transfers-cuda-cc/
    // cudaStream_t stream[sub_m];
    // for (i = 0; i < sub_m; i++) {
    //     cudaStreamCreate(stream + i);
    // }

    cudaMallocPitch((void **) &a_sub_d, &in_pitch, sizeof(double) * sub_k, sub_m);
    cudaMallocPitch((void **) &b_sub_d, &in_pitch, sizeof(double) * sub_n, sub_k);
    cudaMallocPitch((void **) &c_sub_f, &out_pitch, sizeof(float) * sub_n, sub_m);
    // printf("in pitch size: %lu\n", in_pitch);    
    // printf("out pitch size: %lu\n", out_pitch);


    for (i = 0; i < x; i += sub_m) {
        for (j = 0; j < y; j += sub_n) {
            cudaMemset(c_sub_f, 0, sub_m * sub_n * sizeof(float));
            for (k = 0; k < z; k += sub_k) {
                gettimeofday(&h_start, NULL);
                cudaMemcpy2D(a_sub_d, in_pitch, (a + i * y + k), z * sizeof(double), sizeof(double) * sub_k, sub_m, cudaMemcpyHostToDevice);
                // for (ii = i, i_idx = 0; ii < (i + sub_m); ii++, i_idx++) {
                //     cudaMemcpyAsync((a_sub_d + i_idx * sub_n), (a + ii*y + k), sub_k * sizeof(double), cudaMemcpyHostToDevice, stream[i_idx]);
                // }
                cudaMemcpy2D(b_sub_d, in_pitch, (b + k * y + j), y * sizeof(double), sizeof(double) * sub_n, sub_k, cudaMemcpyHostToDevice);

                // for (kk = k, k_idx = 0; kk < (k + sub_k); kk++, k_idx++) {
                //     cudaMemcpyAsync((b_sub_d + k_idx * sub_n), (b + kk * y + j), sub_n * sizeof(double), cudaMemcpyHostToDevice, stream[k_idx]);
                // }
                gettimeofday(&h_end, NULL);
                h2d_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);            
                // cublasDgemm EXPLANATION ------------------------------------------------
                // the memory layout is different from we know
                // a = [0 1; b = [3 2; 
                //      2 3]      1 0]
                // if use a_d then b_d, c[0][0] will be a[0, 0] * b[0, 0] + a[1, 0] * b[0, 1] = 4
                // with b_d then a_d, c[0][0] will be a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] = 1
                // maybe that's because inside GPU it uses column major storage.
                gettimeofday(&h_start, NULL);
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, sub_k, &alpha, b_sub_d, Btype, sub_k, a_sub_d, Atype, sub_m, &beta, c_sub_f, Ctype, sub_m, computetype, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                cudaDeviceSynchronize();
                gettimeofday(&h_end, NULL);
                kernel_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);            
            }
            gettimeofday(&h_start, NULL);
            cudaMemcpy2D((c + i * y + j), x * sizeof(float), c_sub_f, out_pitch, sizeof(float) * sub_n, sub_m, cudaMemcpyDeviceToHost);

            // for (ii = i, i_idx = 0; ii < (i + sub_n); ii++, i_idx++) {
            //     cudaMemcpyAsync((c + ii * y + j), (c_sub_f + i_idx * sub_n), sub_n * sizeof(float), cudaMemcpyDeviceToHost, stream[i_idx]);
            // }   
            gettimeofday(&h_end, NULL);
            d2h_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);              
        }
    }  
    
    cublasDestroy(handle);
    
    // for (i = 0; i < sub_m; i++) {
    //     cudaStreamDestroy(stream[i]);
    // }

    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(c_sub_f);
    printf("h2d time: %f ms\n", (float) h2d_time / 1000);
    printf("kernel time: %f ms\n", (float) kernel_time / 1000);
    printf("d2h time: %f ms\n", (float) d2h_time / 1000);
}

void sequential_blockSgemm_half(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, 
    const double *a, const double *b, float *c) {
    sequential_blockGemmEx(x, y, z, sub_m, sub_n, sub_k, a, b, c, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F);
}

void sequential_blockSgemm(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, 
    const double *a, const double *b, float *c) {
    sequential_blockGemmEx(x, y, z, sub_m, sub_n, sub_k, a, b, c, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F);
}

void sequential_blockDgemm(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, 
    const double *a, const double *b, float *c) {
    size_t i, j, k, ii, kk, i_idx, k_idx;
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

    size_t dsize = sub_m * sub_n;

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
}

// DON'T USE. Lose precision somewhere.
float* wholeMatrixHgemm(size_t m, size_t n, size_t k, const double *a, const double *b, float *c) {
    half alpha = 1.0;
    half beta = 0.0;
    double *a_d, *b_d;
    half *a_h, *b_h, *c_h;
    float *c_f;
    size_t dsize = m * n;

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

void wholeMatrix_GemmEx(size_t m, size_t n, size_t k, const double *a, const double *b, float *c, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t computetype) {
    float alpha = 1.0;
    float beta = 0.0;
    double *a_d, *b_d;
    float *c_f;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    cudaMalloc((void **) &a_d, sizeof(double) * m * k);
    cudaMalloc((void **) &b_d, sizeof(double) * k * n);
    cudaMalloc((void **) &c_f, sizeof(float) * m * n);

    cudaMemcpy(a_d, a, sizeof(double) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(double) * k * n, cudaMemcpyHostToDevice);

    // cublasDgemm EXPLANATION ------------------------------------------------
    // the memory layout is different from we know
    // a = [0 1; b = [3 2; 
    //      2 3]      1 0]
    // if use a_d then b_d, c[0][0] will be a[0, 0] * b[0, 0] + a[1, 0] * b[0, 1] = 4
    // with b_d then a_d, c[0][0] will be a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] = 1
    // maybe that's because inside GPU it uses column major storage.
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, b_f, CUDA_R_16F, k, a_f, CUDA_R_16F, m, &beta, c_f, CUDA_R_32F, m);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, b_d, Btype, k, a_d, Atype, m, &beta, c_f, Ctype, m, computetype, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaFree(a_d);
    cudaFree(b_d);

    cudaMemcpy(c, c_f, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    
    cublasDestroy(handle);

    cudaFree(c_f);
}

void wholeMatrix_Sgemm_half(size_t m, size_t n, size_t k, const double *a, const double *b, float *c) {
    wholeMatrix_GemmEx(m, n, k, a, b, c, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F);
}

void wholeMatrix_Sgemm(size_t m, size_t n, size_t k, const double *a, const double *b, float *c) {
    wholeMatrix_GemmEx(m, n, k, a, b, c, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F);
}

void wholeMatrix_Dgemm(size_t m, size_t n, size_t k, const double *a, const double *b, float *c) {
    double alpha = 1.0;
    double beta = 0.0;
    double *a_d, *b_d, *c_d;
    float *c_f;
    size_t dsize = m * n;

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
}