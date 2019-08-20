extern "C" {
    #include "spdkrpc.h"
}

#include <stdio.h>
#include <stdlib.h>
#include "cublasGEMM.h"

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// for timing
#include <sys/time.h>

// for checking nan
#include <math.h>

#define THREADS_PER_BLOCK 256

__global__ void d2h_kernel(const double *din, half *dout, size_t dsize) {
	size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
	if (idx < dsize)
	{
		dout[idx] = din[idx];
	}
}

int verify(const float *C, const float *answer, int m, int n) {
    // also need to consider floating point error
    const float relativeTolerance = 1e-3;
    int row, col;
    float relativeError;
    for(row = 0; row < m; ++row) {
        for(col = 0; col < n; ++col) {
            if (isnan(C[row*n + col])) {
                printf("(%d, %d) is NaN\n", row, col);
                return 0; 
            }

            if (isinf(C[row*n + col])) {
                printf("(%d, %d) is inf\n", row, col);
                return 0; 
            }
            relativeError = (answer[row*n + col] - C[row*n + col]) / answer[row*n + col];
            if (fabs(relativeError) > relativeTolerance) {
                printf("(%d, %d) = %f, supposed to be %f\n", row, col, C[row*n + col], answer[row*n + col]); 
                printf("TEST FAILED\n\n");
                return 0;
            }    
        }
    }
    printf("TEST PASSED\n\n");
    return 1;
}

int cudaMemcpyFromMmap(struct JSONRPCClient *client, double *dst, const double *src, int id, int x, int y) {
    size_t return_size; 

    return_size = tensorstore_request_submatrix(client, id, x, y);
    if (return_size == 0) {
        return -1;
    }
    cudaMemcpy(dst, src, return_size, cudaMemcpyHostToDevice);
    return 0;
}

int spdk_blockSgemm_half(int request_id, int m, int sub_m, float *c) {
    int i, j, k;
    double *hugepage_addr;
    double *out_ptr;
    struct JSONRPCClient client;
    size_t return_size; 
    int rc;
    double *a_sub_d, *b_sub_d;
    half *a_sub_h, *b_sub_h;
    float *c_sub_f;

    float alpha = 1.0;
    float beta = 1.0;
    struct timeval h_start, h_end;
    unsigned long long fetch_time = 0, transpose_time = 0;
    
    size_t out_pitch, ldc;
    size_t dsize;
    cublasHandle_t handle;

    // initialization
    printf("create cublas handle\n");
    cublasCreate(&handle);
    printf("create cublas handle\n");
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    printf("connect to spdk\n");
    rc = connect_to_spdkrpc_server(&client);
    if (rc) {
        printf("cannot create conntection to SPDK RPC server");
        return rc;
    }

    printf("map to shared page\n");
    hugepage_addr = (double *) mmap_to_tensorstore_hugepage();
    if (hugepage_addr == NULL) {
        return -1;
    }

    // cuda malloc
    printf("allocate memory\n");
    dsize = sub_m * sub_m;
    cudaMalloc((void **) &a_sub_d, sizeof(double) * dsize);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * dsize);
    cudaMalloc((void **) &a_sub_h, sizeof(half) * dsize);
    cudaMalloc((void **) &b_sub_h, sizeof(half) * dsize);

    cudaMallocPitch((void **) &c_sub_f, &out_pitch, sizeof(float) * sub_m, sub_m);
    ldc = out_pitch / sizeof(float);

    printf("doing GEMM\n");
    // blockGEMM
    for (i = 0; i < m / sub_m; i++) {
        for (j = 0; j < m / sub_m; j++) {
            for (k = 0; k < m / sub_m; k++) {
                printf("i: %d, j: %d, k: %d\n", i, j, k);
                // memset(hugepage_addr, 0, HUGEPAGE_SZ);
                gettimeofday(&h_start, NULL);
                cudaMemcpyFromMmap(&client, a_sub_d, hugepage_addr, 0, k, i);
                cudaMemcpyFromMmap(&client, b_sub_d, hugepage_addr, 0, j, k);
                gettimeofday(&h_end, NULL);
                fetch_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   
                d2h_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(a_sub_d, a_sub_h, dsize);
                d2h_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(b_sub_d, b_sub_h, dsize);
                // gemm
                gettimeofday(&h_start, NULL);
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_m, sub_m, &alpha, b_sub_h, CUDA_R_16F, ldc, a_sub_h, CUDA_R_16F, ldc, &beta, c_sub_f, CUDA_R_32F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                cudaDeviceSynchronize();
                gettimeofday(&h_end, NULL);
                transpose_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   
            }
        }   
        cudaMemcpy2D((c + (i * sub_m) * m + (j * sub_m)), m * sizeof(float), c_sub_f, out_pitch, sizeof(float) * sub_m, sub_m, cudaMemcpyDeviceToHost);
    }
    printf("data fetch time: %f ms\n", (float) fetch_time / 1000);
    printf("transpose time: %f ms\n", (float) transpose_time / 1000);

    munmap(hugepage_addr, HUGEPAGE_SZ);
    cublasDestroy(handle);
    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(c_sub_f);
    return 0;
}

int spdk_blockSgemm_half_order(int request_id, int m, int sub_m, float *c) {
    int i, j, k;
    double *hugepage_addr;
    double *out_ptr;
    struct JSONRPCClient client;
    size_t return_size; 
    int rc;
    double *a_sub_d, *b_sub_d;
    half *a_sub_h, *b_sub_h;
    float *c_sub_f;

    float alpha = 1.0;
    float beta = 1.0;
    struct timeval h_start, h_end;
    unsigned long long fetch_time = 0, transpose_time = 0;
    
    size_t out_pitch, ldc;
    size_t dsize;
    cublasHandle_t handle;

    // initialization
    printf("create cublas handle\n");
    cublasCreate(&handle);
    printf("create cublas handle\n");
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    printf("connect to spdk\n");
    rc = connect_to_spdkrpc_server(&client);
    if (rc) {
        printf("cannot create conntection to SPDK RPC server");
        return rc;
    }

    printf("map to shared page\n");
    hugepage_addr = (double *) mmap_to_tensorstore_hugepage();
    if (hugepage_addr == NULL) {
        return -1;
    }

    // cuda malloc
    printf("allocate memory\n");
    dsize = sub_m * sub_m;
    cudaMalloc((void **) &a_sub_d, sizeof(double) * dsize);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * dsize);
    cudaMalloc((void **) &a_sub_h, sizeof(half) * dsize);
    cudaMalloc((void **) &b_sub_h, sizeof(half) * dsize);

    cudaMallocPitch((void **) &c_sub_f, &out_pitch, sizeof(float) * sub_m, sub_m);
    ldc = out_pitch / sizeof(float);

    printf("doing GEMM\n");
    // blockGEMM
    for (k = 0; k < m / sub_m; k++) {
        for (i = 0; i < m / sub_m; i++) {
            gettimeofday(&h_start, NULL);
            cudaMemcpyFromMmap(&client, a_sub_d, hugepage_addr, 0, k, i);
            gettimeofday(&h_end, NULL);
            fetch_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);  
            d2h_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(a_sub_d, a_sub_h, dsize); 
            for (j = 0; j < m / sub_m; j++) {
                printf("i: %d, j: %d, k: %d\n", i, j, k);
                // memset(hugepage_addr, 0, HUGEPAGE_SZ);
                gettimeofday(&h_start, NULL);
                cudaMemcpyFromMmap(&client, b_sub_d, hugepage_addr, 0, j, k);
                gettimeofday(&h_end, NULL);
                fetch_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   
                d2h_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(b_sub_d, b_sub_h, dsize);
                // gemm
                cudaMemcpy2D(c_sub_f, m * sizeof(float), (c + (i * sub_m) * m + (j * sub_m)), out_pitch, sizeof(float) * sub_m, sub_m, cudaMemcpyHostToDevice);
                gettimeofday(&h_start, NULL);
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_m, sub_m, &alpha, b_sub_h, CUDA_R_16F, ldc, a_sub_h, CUDA_R_16F, ldc, &beta, c_sub_f, CUDA_R_32F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                cudaDeviceSynchronize();
                gettimeofday(&h_end, NULL);
                transpose_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   
                cudaMemcpy2D((c + (i * sub_m) * m + (j * sub_m)), m * sizeof(float), c_sub_f, out_pitch, sizeof(float) * sub_m, sub_m, cudaMemcpyDeviceToHost);
            }
        }   
    }
    printf("data fetch time: %f ms\n", (float) fetch_time / 1000);
    printf("transpose time: %f ms\n", (float) transpose_time / 1000);

    munmap(hugepage_addr, HUGEPAGE_SZ);
    cublasDestroy(handle);
    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(c_sub_f);
    return 0;
}


int main(int argc, char** argv) {
    double *a, *b;
    int a_fd, b_fd;
    float *c, *answer_c;
    int is_passed = 0;
    struct timeval h_start, h_end;
    long duration;
    int request_id, n, sub_n;
    
    if (argc < 5) {
        printf("usage: %s <req matrix id> <n> <sub_n> <validated matrix>\n", argv[0]);
        exit(1);
    }

    request_id = atoi(argv[1]);
    n = atoi(argv[2]);
    sub_n = atoi(argv[3]);

    a_fd = open(argv[4], O_RDONLY);
    b_fd = open(argv[4], O_RDONLY);

    a = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, a_fd, 0);
    b = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, b_fd, 0);

    // GEMM configuration
    c = (float *) calloc(n * n, sizeof(float));
    answer_c = (float *) calloc(n * n, sizeof(float));

    // TODO: load the validation matrix operands
    printf("calculating the answer...\n");
    memset(answer_c, 0, n * n * sizeof(float));
    gettimeofday(&h_start, NULL);
    wholeMatrix_Dgemm(n, n, n, a, b, answer_c);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("sequential format GEMM duration: %f ms\n", (float) duration / 1000);    

    printf("calculating the result of the sequential format\n");
    memset(c, 0, n * n * sizeof(float));
    gettimeofday(&h_start, NULL);
    spdk_blockSgemm_half_order(request_id, n, sub_n, c);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("GEMM duration: %f ms\n", (float) duration / 1000);    

    is_passed = verify(c, answer_c, n, n);

    // if (is_passed && need_output) {
    //     char filename[64];
    //     FILE *fptr;
    //     sprintf(filename, "ans_%d.bin", n);
    //     printf("writing sequential answer to %s\n", &filename[0]);
    //     fptr = fopen(filename, "wb");
    //     fwrite(c, sizeof(float), n * n, fptr);
    // }

#ifdef DEBUG
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", a[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", b[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", c[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
#endif

    munmap(a, sizeof(double) * n * n);
    munmap(b, sizeof(double) * n * n);
    close(a_fd);
    close(b_fd);

    free(answer_c);
    free(c);

    return 0;
}