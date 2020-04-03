#include "cublasGEMM.h"

#define MAX_STREAMS 1

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

// pitched memory address calculation.
// T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
// float* element = (float*)((char*)devPtr + r * pitch + c * sizeof(float));
__global__ void d2f_kernel_pitch_2D(const double *din, const size_t in_pitch, float *dout, const size_t out_pitch, const size_t nrows, const size_t ncols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if ((idx < ncols) && (idy < nrows))
	{
        double *in = (double *)((char*) din + idy * in_pitch); // in_pitch = 512
        float *out = (float *)((char*) dout + idy * out_pitch);
	    out[idx] = (float) in[idx];
	}
}

__global__ void d2h_kernel_pitch_2D(const double *din, const size_t in_pitch, half *dout, const size_t out_pitch, const size_t nrows, const size_t ncols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if ((idx < ncols) && (idy < nrows))
	{
        double *in = (double *)((char*) din + idy * in_pitch);
        half *out = (half *)((char*) dout + idy * out_pitch);
	    out[idx] = (half) in[idx];
    }
}

__global__ void d2f_kernel_pitch(const double *din, const size_t in_pitch, float *dout, const size_t out_pitch, const size_t dsize, const size_t ncols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < dsize)
	{
        size_t in_i = idx / ncols;
        size_t in_j = idx % ncols;

        size_t out_i = idx / ncols;
        size_t out_j = idx % ncols;
        double *in = (double *)((char*) din + in_i * in_pitch); // in_pitch = 512
        float *out = (float *)((char*) dout + out_i * out_pitch);

        out[out_j] = in[in_j];
        // printf("%d: %lu %lu %lu %lu, %f %f\n", idx, in_i, in_j, out_i, out_j, in[out_j], out[in_j]);
	}
}

__global__ void h2f_kernel(half *din, float *dout, size_t dsize) {
	size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
	if (idx < dsize)
	{
		dout[idx] = din[idx];
	}
}

__global__ void reduction_kernel_2D(float *in, size_t sub_m, size_t sub_n, size_t num_streams) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int i, res = 0;

	if ((idx < sub_n) && (idy < sub_m))
	{
        float *ptr = in + idy * sub_n + idx;
        for (i = 0; i < num_streams; i++) {
            res += ptr[i * sub_m * sub_n];
        }
        in[idy * sub_n + idx] = res;
	}
}

__global__ void reduction_kernel(float *in, size_t sub_m, size_t sub_n, size_t num_streams) {
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
    int i, res = 0;
    if (idx < sub_m * sub_n) {
        for (i = 0; i < num_streams; i++) {
            res += in[idx + i * sub_m * sub_n];
        }
        in[idx] = res;
    }
}

void tensor_blockGemmEx_async_v3(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, 
    const double *a, const double *b, float *c, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t computetype) {
    size_t i, j, k;
    size_t cross_row = x * sub_k, cross_col = sub_m * sub_k;
    float alpha = 1.0;
    float beta = 1.0;

    struct timeval h_start, h_end;
    unsigned long long h2d_time = 0, d2h_time = 0, kernel_time = 0, reduction_time = 0;

    // think about a way to use the power of 2 as the number of streams
    size_t memory_usage = sizeof(double) * sub_m * sub_k + sizeof(double) * sub_k * sub_n + sizeof(float) * sub_m * sub_n;
    size_t num_streams = 32;
    num_streams = (num_streams < ((size_t) 8192 * 1048576 / (memory_usage))) ? num_streams : ((size_t) 8192 * 1048576 / (memory_usage));
    printf("num_streams: %lu\n", num_streams);
    size_t num_outstreams = 4;
    size_t num_instreams = 8;
    size_t thread_num = 1024;
    cublasHandle_t handle[num_streams];
    cudaStream_t stream[num_streams];
    for (i = 0; i < num_streams; i++) {
        cublasCreate(handle + i);
        cudaStreamCreate(stream + i);
        cublasSetMathMode(handle[i], CUBLAS_TENSOR_OP_MATH);
        cublasSetStream(handle[i], stream[i]);
    }

    int out_stream_index, in_stream_index;
    double *a_sub_d, *b_sub_d;
    float *c_sub_f;
    // here cannot exceed the GPU memory
    cudaMalloc((void **) &a_sub_d, sizeof(double) * sub_m * sub_k * num_streams);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * sub_k * sub_n * num_streams);
    cudaMalloc((void **) &c_sub_f, sizeof(float) * sub_m * sub_n * num_streams);

    // custom block gemm
    for (i = 0; i < (x / sub_m); i++) {
        for (j = 0; j < (y / sub_n); j++) {
            out_stream_index = (j % num_outstreams) * num_instreams;
            cudaMemsetAsync(c_sub_f + out_stream_index * sub_m * sub_n, 0, sizeof(float) * sub_m * sub_n * num_instreams, stream[out_stream_index]);
            for (k = 0; k < (z / sub_k); k++) {
                // here we can use GPUDirect?
                in_stream_index = (k % num_instreams) + out_stream_index;
                gettimeofday(&h_start, NULL);
                cudaMemcpyAsync(a_sub_d + in_stream_index * sub_m * sub_k, (a + i * cross_row + k * cross_col), sub_m * sub_k * sizeof(double), cudaMemcpyHostToDevice, stream[in_stream_index]);    
                cudaMemcpyAsync(b_sub_d + in_stream_index * sub_k * sub_n, (b + k * cross_row + j * cross_col), sub_k * sub_n * sizeof(double), cudaMemcpyHostToDevice, stream[in_stream_index]);
                gettimeofday(&h_end, NULL);
                h2d_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);            
                // async execution (ref: https://forums.developer.nvidia.com/t/async-cublas/2837)
                // cudaDataType_t helps users to convert data inside the function call
                
                gettimeofday(&h_start, NULL);
                cublasGemmEx(handle[in_stream_index], CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, sub_k, &alpha, b_sub_d + in_stream_index * sub_k * sub_n, Btype, sub_k, a_sub_d + in_stream_index * sub_m * sub_k, Atype, sub_m, &beta, c_sub_f + in_stream_index * sub_m * sub_n, Ctype, sub_m, computetype, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                gettimeofday(&h_end, NULL);
                kernel_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);            
            }

            for (k = out_stream_index; k < num_instreams; k++) {
                cudaStreamSynchronize(stream[k]);
            }

            if (num_streams > 1) {
                gettimeofday(&h_start, NULL);
                reduction_kernel<<<(sub_m*sub_n+thread_num-1)/thread_num, thread_num, 0, stream[out_stream_index]>>>(c_sub_f + out_stream_index * sub_m * sub_n, sub_m, sub_n, num_streams);
                gettimeofday(&h_end, NULL);
                reduction_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);                
            }
            gettimeofday(&h_start, NULL);
            cudaMemcpyAsync((c + i * cross_row + j * cross_col), (c_sub_f + out_stream_index * sub_m * sub_n), sub_m * sub_n * sizeof(float), cudaMemcpyDeviceToHost, stream[out_stream_index]);
            gettimeofday(&h_end, NULL);
            d2h_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);            
        }
    }

    cudaDeviceSynchronize();
    for (i = 0; i < num_streams; i++) {
        cublasDestroy(handle[i]);
        cudaStreamDestroy(stream[i]);
    }
    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(c_sub_f);
    printf("h2d time: %f ms\n", (float) h2d_time / 1000);
    printf("kernel time: %f ms\n", (float) kernel_time / 1000);
    printf("reduction time: %f ms\n", (float) reduction_time / 1000);
    printf("d2h time: %f ms\n", (float) d2h_time / 1000);
}

void tensor_blockGemmEx_async_v2(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, 
    const double *a, const double *b, float *c, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t computetype) {
    size_t i, j, k;
    size_t cross_row = x * sub_k, cross_col = sub_m * sub_k;
    float alpha = 1.0;
    float beta = 1.0;
    double *a_sub_d, *b_sub_d;
    float *c_sub_f;
    struct timeval h_start, h_end;
    unsigned long long h2d_time = 0, d2h_time = 0, kernel_time = 0, reduction_time = 0;

    // think about a way to use the power of 2 as the number of streams
    size_t memory_usage = sizeof(double) * sub_m * sub_k + sizeof(double) * sub_k * sub_n + sizeof(float) * sub_m * sub_n;
    size_t num_streams = (MAX_STREAMS < ((size_t) 8192 * 1048576 / (memory_usage))) ? MAX_STREAMS : ((size_t) 8192 * 1048576 / (memory_usage));
    printf("num_streams: %lu\n", num_streams);

    cublasHandle_t handle[num_streams];
    cudaStream_t stream[num_streams];
    for (i = 0; i < num_streams; i++) {
        cublasCreate(handle + i);
        cudaStreamCreate(stream + i);
        cublasSetMathMode(handle[i], CUBLAS_TENSOR_OP_MATH);
        cublasSetStream(handle[i], stream[i]);
    }

    int stream_index;
    
    // here cannot exceed the GPU memory
    cudaMalloc((void **) &a_sub_d, sizeof(double) * sub_m * sub_k * num_streams);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * sub_k * sub_n * num_streams);
    cudaMalloc((void **) &c_sub_f, sizeof(float) * sub_m * sub_n * num_streams);

    // custom block gemm
    for (i = 0; i < (x / sub_m); i++) {
        for (j = 0; j < (y / sub_n); j++) {
            stream_index = j % num_streams;
            cudaMemsetAsync(c_sub_f + stream_index * sub_m * sub_n, 0, sizeof(float) * sub_m * sub_n, stream[stream_index]);
            for (k = 0; k < (z / sub_k); k++) {
                // here we can use GPUDirect?
                gettimeofday(&h_start, NULL);
                cudaMemcpyAsync(a_sub_d + stream_index * sub_m * sub_k, (a + i * cross_row + k * cross_col), sub_m * sub_k * sizeof(double), cudaMemcpyHostToDevice, stream[stream_index]);    
                cudaMemcpyAsync(b_sub_d + stream_index * sub_k * sub_n, (b + k * cross_row + j * cross_col), sub_k * sub_n * sizeof(double), cudaMemcpyHostToDevice, stream[stream_index]);
                gettimeofday(&h_end, NULL);
                h2d_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);            
                // async execution (ref: https://forums.developer.nvidia.com/t/async-cublas/2837)
                // cudaDataType_t helps users to convert data inside the function call
                
                gettimeofday(&h_start, NULL);
                cublasGemmEx(handle[stream_index], CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, sub_k, &alpha, b_sub_d + stream_index * sub_k * sub_n, Btype, sub_k, a_sub_d + stream_index * sub_m * sub_k, Atype, sub_m, &beta, c_sub_f + stream_index * sub_m * sub_n, Ctype, sub_m, computetype, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                gettimeofday(&h_end, NULL);
                kernel_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);            
            }
            // cudaDeviceSynchronize();
            // if (num_streams > 1) {
            //     gettimeofday(&h_start, NULL);
            //     reduction_kernel<<<(sub_m*sub_n+thread_num-1)/thread_num, thread_num>>>(c_sub_f, sub_m, sub_n, num_streams);
            //     gettimeofday(&h_end, NULL);
            //     reduction_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);                
            // }
            gettimeofday(&h_start, NULL);
            cudaMemcpyAsync((c + i * cross_row + j * cross_col), (c_sub_f + stream_index * sub_m * sub_n), sub_m * sub_n * sizeof(float), cudaMemcpyDeviceToHost, stream[stream_index]);
            gettimeofday(&h_end, NULL);
            d2h_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);            
        }
    }
    for (i = 0; i < x; i++) {
        for (j = 0; j < y; j++) {
            printf("%f ", c[i*y+j]);
        }
        printf("\n");
    }  
    printf("\n");

    cudaDeviceSynchronize();
    for (i = 0; i < num_streams; i++) {
        cublasDestroy(handle[i]);
        cudaStreamDestroy(stream[i]);
    }
    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(c_sub_f);
    printf("h2d time: %f ms\n", (float) h2d_time / 1000);
    printf("kernel time: %f ms\n", (float) kernel_time / 1000);
    printf("reduction time: %f ms\n", (float) reduction_time / 1000);
    printf("d2h time: %f ms\n", (float) d2h_time / 1000);
}

// inner loop runs asynchronously
// assume the inputs are half precision now
void tensor_blockGemmEx_async(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, 
    const double *a, const double *b, float *c, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t computetype) {
    size_t i, j, k;
    size_t cross_row = x * sub_k, cross_col = sub_m * sub_k;
    float alpha = 1.0;
    float beta = 1.0;
    half *a_sub_d, *b_sub_d;
    double *temp_a, *temp_b;
    float *c_sub_f;
    struct timeval h_start, h_end;
    unsigned long long h2d_time = 0, d2h_time = 0, kernel_time = 0, reduction_time = 0;

    size_t dsize = sub_m * sub_n;
    // think about a way to use the power of 2 as the number of streams
    size_t memory_usage = sizeof(double) * sub_m * sub_k + sizeof(double) * sub_k * sub_n + sizeof(float) * sub_m * sub_n;
    size_t num_streams = (MAX_STREAMS < ((size_t) 8192 * 1048576 / (memory_usage))) ? MAX_STREAMS : ((size_t) 8192 * 1048576 / (memory_usage));
    printf("num_streams: %lu\n", num_streams);

    cudaMalloc((void **) &a_sub_d, sizeof(half) * num_streams * sub_m * sub_k);
    cudaMalloc((void **) &b_sub_d, sizeof(half) * num_streams * sub_k * sub_n);
    cudaMalloc((void **) &c_sub_f, sizeof(float) * num_streams * sub_k * sub_n);    
    cudaMalloc((void **) &temp_a, sizeof(double) * num_streams * sub_m * sub_k);
    cudaMalloc((void **) &temp_b, sizeof(double) * num_streams * sub_k * sub_n);    

    cublasHandle_t handle[num_streams];
    cudaStream_t stream[num_streams];
    for (i = 0; i < num_streams; i++) {
        cublasCreate(handle + i);
        cudaStreamCreate(stream + i);
        cublasSetMathMode(handle[i], CUBLAS_TENSOR_OP_MATH);
        cublasSetStream(handle[i], stream[i]);
    }

    // custom block gemm
    int stream_index;
    dim3 gridSize((sub_m+32-1)/32, (sub_n+32-1)/32);
    dim3 blockSize(32, 32);
    for (i = 0; i < (x / sub_m); i++) {
        for (j = 0; j < (y / sub_n); j++) {
            cudaMemset(c_sub_f, 0, sizeof(float) * sub_m * sub_n * num_streams);
            for (k = 0; k < (z / sub_k); k++) {
                stream_index = k % num_streams;
                // here we can use GPUDirect?
                gettimeofday(&h_start, NULL);
                cudaMemcpyAsync(temp_a + stream_index * sub_m * sub_k, (a + i * cross_row + k * cross_col), sub_m * sub_k * sizeof(double), cudaMemcpyHostToDevice, stream[stream_index]);   
                d2h_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK, 0, stream[stream_index]>>>(temp_a + stream_index * sub_m * sub_k, a_sub_d + stream_index * sub_m * sub_k, dsize);
                cudaMemcpyAsync(temp_b + stream_index * sub_k * sub_n, (b + k * cross_row + j * cross_col), sub_k * sub_n * sizeof(double), cudaMemcpyHostToDevice, stream[stream_index]);
                d2h_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK, 0, stream[stream_index]>>>(temp_b + stream_index * sub_m * sub_k, b_sub_d + stream_index * sub_m * sub_k, dsize);
                gettimeofday(&h_end, NULL);
                h2d_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);            
                // async execution (ref: https://forums.developer.nvidia.com/t/async-cublas/2837)
                // cudaDataType_t helps users to convert data inside the function call
                gettimeofday(&h_start, NULL);
                cublasGemmEx(handle[stream_index], CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, sub_k, &alpha, b_sub_d + stream_index * sub_m * sub_n, Btype, sub_k, a_sub_d + stream_index * sub_m * sub_n, Atype, sub_m, &beta, c_sub_f + stream_index * sub_m * sub_n, Ctype, sub_m, computetype, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                gettimeofday(&h_end, NULL);
                kernel_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);            
            }
            cudaDeviceSynchronize();
            if (num_streams > 1) {
                gettimeofday(&h_start, NULL);
                reduction_kernel_2D<<<gridSize, blockSize>>>(c_sub_f, sub_m, sub_n, num_streams);
                gettimeofday(&h_end, NULL);
                reduction_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);                
            }

            gettimeofday(&h_start, NULL);
            cudaMemcpy((c + i * cross_row + j * cross_col), c_sub_f, sub_m * sub_n * sizeof(float), cudaMemcpyDeviceToHost);
            gettimeofday(&h_end, NULL);
            d2h_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);            
        }
    }

    // for (i = 0; i < x; i++) {
    //     for (j = 0; j < y; j++) {
    //         printf("%f ", c[i*y+j]);
    //     }
    //     printf("\n");
    // }  
    // printf("\n");

    for (i = 0; i < num_streams; i++) {
        cublasDestroy(handle[i]);
        cudaStreamDestroy(stream[i]);
    }

    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(c_sub_f);
    cudaFree(temp_a);
    cudaFree(temp_b);


    printf("h2d time: %f ms\n", (float) h2d_time / 1000);
    printf("kernel time: %f ms\n", (float) kernel_time / 1000);
    printf("reduction time: %f ms\n", (float) reduction_time / 1000);
    printf("d2h time: %f ms\n", (float) d2h_time / 1000);
}

void tensor_blockGemmEx(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, 
    const double *a, const double *b, float *c, cudaDataType_t Atype, cudaDataType_t Btype, cudaDataType_t Ctype, cudaDataType_t computetype) {
    size_t i, j, k;
    size_t cross_row = x * sub_k, cross_col = sub_m * sub_k;
    float alpha = 1.0;
    float beta = 1.0;
    void *a_sub_d, *b_sub_d;
    double *temp_a, *temp_b;

    float *c_sub_f;
    struct timeval h_start, h_end;
    // assume input/output arrays are the same size and square matrix now
    const size_t dsize = sub_m * sub_n;

    unsigned long long h2d_time = 0, d2h_time = 0, kernel_time = 0;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    // cudaStream_t stream[z / sub_k];
    // for (i = 0; i < z / sub_k; i++) {
    //     cudaStreamCreate(stream + i);
    // }
    

    if (Atype == CUDA_R_16F || Atype == CUDA_R_32F) {
        cudaMalloc((void **) &temp_a, sizeof(double) * sub_m * sub_k);
        cudaMalloc((void **) &temp_b, sizeof(double) * sub_k * sub_n);    
        if (Atype == CUDA_R_16F) {
            // printf("half type\n");
            cudaMalloc((void **) &a_sub_d, sizeof(half) * sub_m * sub_k);
            cudaMalloc((void **) &b_sub_d, sizeof(half) * sub_k * sub_n);
        } else {
            // printf("float type\n");
            cudaMalloc((void **) &a_sub_d, sizeof(float) * sub_m * sub_k);
            cudaMalloc((void **) &b_sub_d, sizeof(float) * sub_k * sub_n);
        }
    } else if (Atype == CUDA_R_64F) {
        cudaMalloc((void **) &a_sub_d, sizeof(double) * sub_m * sub_k);
        cudaMalloc((void **) &b_sub_d, sizeof(double) * sub_k * sub_n);    
    } else {
        printf("input type: %d is not supported\n", Atype);
        return;
    }

    cudaMalloc((void **) &c_sub_f, sizeof(float) * sub_m * sub_n);

    // custom block gemm
    for (i = 0; i < (x / sub_m); i++) {
        for (j = 0; j < (y / sub_n); j++) {
            cudaMemset(c_sub_f, 0, sub_m * sub_n * sizeof(float));
            for (k = 0; k < (z / sub_k); k++) {
                // here we can use GPUDirect?
                gettimeofday(&h_start, NULL);
                if (Atype == CUDA_R_16F || Atype == CUDA_R_32F) {
                    cudaMemcpy(temp_a, (a + i * cross_row + k * cross_col), sub_m * sub_k * sizeof(double), cudaMemcpyHostToDevice);    
                    cudaMemcpy(temp_b, (b + k * cross_row + j * cross_col), sub_k * sub_n * sizeof(double), cudaMemcpyHostToDevice);
                    if (Atype == CUDA_R_16F) {
                        d2h_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(temp_a, (half *) a_sub_d, dsize);
                        d2h_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(temp_b, (half *) b_sub_d, dsize);
                    } else { 
                        d2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(temp_a, (float *) a_sub_d, dsize);
                        d2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(temp_b, (float *) b_sub_d, dsize);
                    }
                } else { // CUDA_R_64F
                    cudaMemcpy(a_sub_d, (a + i * cross_row + k * cross_col), sub_m * sub_k * sizeof(double), cudaMemcpyHostToDevice);    
                    cudaMemcpy(b_sub_d, (b + k * cross_row + j * cross_col), sub_k * sub_n * sizeof(double), cudaMemcpyHostToDevice);
                } 
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

    if (Atype == CUDA_R_16F || Atype == CUDA_R_32F) {
        cudaFree(temp_a);
        cudaFree(temp_b);
    } 
    
    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(c_sub_f);
    printf("h2d time: %f ms\n", (float) h2d_time / 1000);
    printf("kernel time: %f ms\n", (float) kernel_time / 1000);
    printf("d2h time: %f ms\n", (float) d2h_time / 1000);
}

void tensor_blockSgemm_half_async_v3(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, 
    const double *a, const double *b, float *c) {
    tensor_blockGemmEx_async_v3(x, y, z, sub_m, sub_n, sub_k, a, b, c, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F);
}

void tensor_blockSgemm_half_async_v2(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, 
    const double *a, const double *b, float *c) {
    tensor_blockGemmEx_async_v2(x, y, z, sub_m, sub_n, sub_k, a, b, c, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F);
}

void tensor_blockSgemm_half_async(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, 
    const double *a, const double *b, float *c) {
    tensor_blockGemmEx_async(x, y, z, sub_m, sub_n, sub_k, a, b, c, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F);
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
    void *a_sub_d, *b_sub_d;
    double *temp_a, *temp_b;
    float *c_sub_f;
    struct timeval h_start, h_end;
    unsigned long long h2d_time = 0, d2h_time = 0, kernel_time = 0;
    size_t a_in_pitch, converted_a_in_pitch;
    size_t b_in_pitch, converted_b_in_pitch;
    size_t out_pitch;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    dim3 gridSize((sub_m+32-1)/32, (sub_n+32-1)/32);
    dim3 blockSize(32, 32);

    // for copying data and running kernel asynchronously: https://devblogs.nvidia.com/how-overlap-data-transfers-cuda-cc/
    // cudaStream_t stream[sub_m];
    // for (i = 0; i < sub_m; i++) {
    //     cudaStreamCreate(stream + i);
    // }

    size_t lda;
    size_t ldb;
    size_t ldc;
    if (Atype == CUDA_R_16F || Atype == CUDA_R_32F) {
        cudaMallocPitch((void **) &temp_a, &a_in_pitch, sizeof(double) * sub_k, sub_m);
        cudaMallocPitch((void **) &temp_b, &b_in_pitch, sizeof(double) * sub_n, sub_k);    
        if (Atype == CUDA_R_16F) {
            // cudaMalloc((void **) &a_sub_d, sizeof(half) * sub_m * sub_k);
            // cudaMalloc((void **) &b_sub_d, sizeof(half) * sub_k * sub_n);
            cudaMallocPitch((void **) &a_sub_d, &converted_a_in_pitch, sizeof(half) * sub_k, sub_m);
            cudaMallocPitch((void **) &b_sub_d, &converted_b_in_pitch, sizeof(half) * sub_n, sub_k);
            lda = converted_a_in_pitch / sizeof(half);
            ldb = converted_b_in_pitch / sizeof(half);
        } else {
            // cudaMalloc((void **) &a_sub_d, sizeof(float) * sub_m * sub_k);
            // cudaMalloc((void **) &b_sub_d, sizeof(float) * sub_k * sub_n);
            cudaMallocPitch((void **) &a_sub_d, &converted_a_in_pitch, sizeof(float) * sub_k, sub_m);
            cudaMallocPitch((void **) &b_sub_d, &converted_b_in_pitch, sizeof(float) * sub_n, sub_k);
            lda = converted_a_in_pitch / sizeof(float);
            ldb = converted_b_in_pitch / sizeof(float);
        }
    } else if (Atype == CUDA_R_64F) {
        cudaMallocPitch((void **) &a_sub_d, &a_in_pitch, sizeof(double) * sub_k, sub_m);
        cudaMallocPitch((void **) &b_sub_d, &b_in_pitch, sizeof(double) * sub_n, sub_k);    
        lda = a_in_pitch / sizeof(double);
        ldb = b_in_pitch / sizeof(double);
    } else {
        printf("input type: %d is not supported\n", Atype);
        return;
    }

    cudaMallocPitch((void **) &c_sub_f, &out_pitch, sizeof(float) * sub_n, sub_m);
    ldc = out_pitch / sizeof(float);

    printf("a pitch: %lu, b pitch: %lu\n", a_in_pitch, b_in_pitch);   
    printf("converted a pitch: %lu, b pitch: %lu\n", converted_a_in_pitch, converted_b_in_pitch);
    printf("out pitch size: %lu\n", out_pitch);
    printf("lda: %lu, ldb: %lu, ldc: %lu\n", lda, ldb, ldc);
    // printf("temp_a address: %p\n", temp_a);
    // printf("temp_b address: %p\n", temp_b);

    for (i = 0; i < x; i += sub_m) {
        for (j = 0; j < y; j += sub_n) {
            cudaMemset2D(c_sub_f, out_pitch, 0, sub_n * sizeof(float), sub_m);
            for (k = 0; k < z; k += sub_k) {
                gettimeofday(&h_start, NULL);
                if (Atype == CUDA_R_16F || Atype == CUDA_R_32F) {
                    cudaMemcpy2D(temp_a, a_in_pitch, (a + i * y + k), z * sizeof(double), sizeof(double) * sub_k, sub_m, cudaMemcpyHostToDevice);
                    cudaMemcpy2D(temp_b, b_in_pitch, (b + k * y + j), y * sizeof(double), sizeof(double) * sub_n, sub_k, cudaMemcpyHostToDevice);
                    if (Atype == CUDA_R_16F) {
                        d2h_kernel_pitch_2D<<<gridSize, blockSize>>>(temp_a, a_in_pitch, (half *) a_sub_d, converted_a_in_pitch, sub_m, sub_k);
                        d2h_kernel_pitch_2D<<<gridSize, blockSize>>>(temp_b, b_in_pitch, (half *) b_sub_d, converted_b_in_pitch, sub_k, sub_n);
                    } else { 
                        d2f_kernel_pitch_2D<<<gridSize, blockSize>>>(temp_a, a_in_pitch, (float *) a_sub_d, converted_a_in_pitch, sub_m, sub_k);
                        d2f_kernel_pitch_2D<<<gridSize, blockSize>>>(temp_b, b_in_pitch, (float *) b_sub_d, converted_b_in_pitch, sub_k, sub_n);
                    }
                } 
                else { // CUDA_R_64F
                    cudaMemcpy2D(a_sub_d, a_in_pitch, (a + i * y + k), z * sizeof(double), sizeof(double) * sub_k, sub_m, cudaMemcpyHostToDevice);
                    cudaMemcpy2D(b_sub_d, b_in_pitch, (b + k * y + j), y * sizeof(double), sizeof(double) * sub_n, sub_k, cudaMemcpyHostToDevice);    
                } 
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
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, sub_k, &alpha, b_sub_d, Btype, ldb, a_sub_d, Atype, lda, &beta, c_sub_f, Ctype, ldc, computetype, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                cudaDeviceSynchronize();
                gettimeofday(&h_end, NULL);
                kernel_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);            
            }
            gettimeofday(&h_start, NULL);
            cudaMemcpy2D((c + i * y + j), y * sizeof(float), c_sub_f, out_pitch, sizeof(float) * sub_n, sub_m, cudaMemcpyDeviceToHost);

            // for (ii = i, i_idx = 0; ii < (i + sub_n); ii++, i_idx++) {
            //     cudaMemcpyAsync((c + ii * y + j), (c_sub_f + i_idx * sub_n), sub_n * sizeof(float), cudaMemcpyDeviceToHost, stream[i_idx]);
            // }   
            gettimeofday(&h_end, NULL);
            d2h_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);              
        }
    }  
    
    for (i = 0; i < x; i++) {
        for (j = 0; j < y; j++) {
            printf("%f ", c[i*y+j]);
        }
        printf("\n");
    }  
    printf("\n");
    cublasDestroy(handle);
    
    if (Atype == CUDA_R_16F || Atype == CUDA_R_32F) {
        cudaFree(temp_a);
        cudaFree(temp_b);
    } 

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

void sequential_blockDgemm_2D(size_t x, size_t y, size_t z, size_t sub_m, size_t sub_n, size_t sub_k, 
    const double *a, const double *b, float *c) {
    size_t i, j, k, ii, kk, i_idx, k_idx;
    double alpha = 1.0;
    double beta = 1.0;
    double *a_sub_d, *b_sub_d, *c_sub_d;
    float *c_sub_f;
    size_t a_in_pitch, b_in_pitch, out_f_pitch, out_d_pitch;

    cublasHandle_t handle;
    cublasCreate(&handle);

    // cudaMalloc((void **) &a_sub_d, sizeof(double) * sub_k * sub_m);
    // cudaMalloc((void **) &b_sub_d, sizeof(double) * sub_n * sub_k);
    // cudaMalloc((void **) &c_sub_d, sizeof(double) * sub_n * sub_m);

    // cause reshape
    cudaMallocPitch((void **) &a_sub_d, &a_in_pitch, sizeof(double) * sub_k, sub_m);
    cudaMallocPitch((void **) &b_sub_d, &b_in_pitch, sizeof(double) * sub_n, sub_k);
    cudaMallocPitch((void **) &c_sub_d, &out_d_pitch, sizeof(double) * sub_n, sub_m);
    cudaMallocPitch((void **) &c_sub_f, &out_f_pitch, sizeof(float) * sub_n, sub_m);

    printf("a_in_pitch size: %lu\n", a_in_pitch); 
    printf("b_in_pitch size: %lu\n", b_in_pitch);   
  
    printf("out_d_pitch size: %lu\n", out_d_pitch);
    printf("out_f_pitch size: %lu\n", out_f_pitch);

    size_t dsize = sub_m * sub_n;
    size_t lda = a_in_pitch / sizeof(double);
    size_t ldb = b_in_pitch / sizeof(double);
    size_t ldc_d = out_d_pitch / sizeof(double);
    size_t ldc_f = out_f_pitch / sizeof(float);
    printf("ldc_d: %lu, ldc_f: %lu\n", ldc_d, ldc_f);

    dim3 gridSize((sub_m+32-1)/32, (sub_n+32-1)/32);
    dim3 blockSize(32, 32);
    
    for (i = 0; i < x; i += sub_m) {
        for (j = 0; j < y; j += sub_n) {
            // printf("memset\n");
            cudaMemset2D(c_sub_d, out_d_pitch, 0, sub_n * sizeof(double), sub_m);
            for (k = 0; k < z; k += sub_k) {
                // cudaMemcpy2D(c_sub_d, out_d_pitch, (a + i*y + k), z * sizeof(double), sub_k * sizeof(double), sub_m, cudaMemcpyHostToDevice);
                cudaMemcpy2D(a_sub_d, a_in_pitch, (a + i*y + k), z * sizeof(double), sub_k * sizeof(double), sub_m, cudaMemcpyHostToDevice);
                
                // cudaMemcpy2D(c_sub_d, out_d_pitch, (b + k*y + j), y * sizeof(double), sub_n * sizeof(double), sub_k, cudaMemcpyHostToDevice);
                cudaMemcpy2D(b_sub_d, b_in_pitch, (b + k*y + j), y * sizeof(double), sub_n * sizeof(double), sub_k, cudaMemcpyHostToDevice);

                // cublasDgemm EXPLANATION ------------------------------------------------
                // the memory layout is different from we know
                // a = [0 1; b = [3 2; 
                //      2 3]      1 0]
                // if use a_d then b_d, c[0][0] will be a[0, 0] * b[0, 0] + a[1, 0] * b[0, 1] = 4
                // with b_d then a_d, c[0][0] will be a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] = 1
                // maybe that's because inside GPU it uses column major storage.
                // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, sub_k, &alpha, b_sub_d, sub_k, a_sub_d, sub_m, &beta, c_sub_d, sub_m);
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, sub_k, &alpha, b_sub_d, ldb, a_sub_d, lda, &beta, c_sub_d, ldc_d);
                // cudaMemcpy2D((c_d + i*y + k), y * sizeof(double), c_sub_d, out_d_pitch, sub_n * sizeof(double), sub_m, cudaMemcpyDeviceToHost);
                // cudaMemcpy2D((c_d + k*y + j), y * sizeof(double), c_sub_d, out_d_pitch, sub_n * sizeof(double), sub_m, cudaMemcpyDeviceToHost);
            }
            d2f_kernel_pitch_2D<<<gridSize, blockSize>>>(c_sub_d, out_d_pitch, c_sub_f, out_f_pitch, sub_m, sub_n);
            // d2f_kernel_pitch<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(c_sub_d, out_d_pitch, c_sub_f, out_f_pitch, dsize, sub_n);
            cudaMemcpy2D((c + i*y + j), y * sizeof(float), c_sub_f, out_f_pitch, sub_n * sizeof(float), sub_m, cudaMemcpyDeviceToHost);
        }
    }  

    // for (i = 0; i < x; i++) {
    //     for (j = 0; j < y; j++) {
    //         printf("%f ", c[i*y+j]);
    //     }
    //     printf("\n");
    // }  
    // printf("\n");
    // printf("conversion\n");
    // cudaMemcpy(c, c_f, dsize * sizeof(float), cudaMemcpyDeviceToHost);
    cublasDestroy(handle);

    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(c_sub_d);
    cudaFree(c_sub_f);
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
    void *a_d, *b_d;
    float *c_f;
    // assume input/output arrays are the same size and square matrix now
    const size_t dsize = m * n;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    // Find out we have to make type conversion first.
    // Assume Atype == Btype
    // arrays need to be converted only when input type is half or float 
    if (Atype == CUDA_R_16F || Atype == CUDA_R_32F) {
        double *temp_a, *temp_b;
        cudaMalloc((void **) &temp_a, sizeof(double) * m * k);
        cudaMalloc((void **) &temp_b, sizeof(double) * k * n);
        cudaMemcpy(temp_a, a, sizeof(double) * m * k, cudaMemcpyHostToDevice);
        cudaMemcpy(temp_b, b, sizeof(double) * k * n, cudaMemcpyHostToDevice);  
        if (Atype == CUDA_R_16F) {
            cudaMalloc((void **) &a_d, sizeof(half) * m * k);
            cudaMalloc((void **) &b_d, sizeof(half) * k * n);
            d2h_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(temp_a, (half *) a_d, dsize);
            d2h_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(temp_b, (half *) b_d, dsize);
        } else {
            cudaMalloc((void **) &a_d, sizeof(float) * m * k);
            cudaMalloc((void **) &b_d, sizeof(float) * k * n);    
            d2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(temp_a, (float *) a_d, dsize);
            d2f_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(temp_b, (float *) b_d, dsize);
        }
        cudaFree(temp_a);
        cudaFree(temp_b);
    } else if (Atype == CUDA_R_64F) {
        cudaMemcpy(a_d, a, sizeof(double) * m * k, cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b, sizeof(double) * k * n, cudaMemcpyHostToDevice);    
    } else {
        printf("input type: %d is not supported\n", Atype);
        return;
    }
    
    cudaMalloc((void **) &c_f, sizeof(float) * m * n);

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