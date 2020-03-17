#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

float* tensor_blockmm(int x, int y, int z, int sub_m, int sub_n, int sub_k, 
    double *a, double *b, float *c) {
    int i, j, k;
    int cross_row = x * sub_k, cross_col = sub_m * sub_k;
    double alpha = 1.0;
    double beta = 1.0;
    double *a_sub_d, *b_sub_d, *c_sub_d;
    double *c_h;
    cublasHandle_t handle;
    cublasCreate(&handle);

    c_h = (double *) calloc(x * y, sizeof(double));

    cudaMalloc((void **) &a_sub_d, sizeof(double) * sub_m * sub_k);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * sub_k * sub_n);
    cudaMalloc((void **) &c_sub_d, sizeof(double) * sub_m * sub_n);

    // custom block gemm
    for (i = 0; i < (x / sub_m); i++) {
        for (j = 0; j < (y / sub_n); j++) {
            cudaMemcpy(c_sub_d, (c_h + i * cross_row + j * cross_col), sub_m * sub_n * sizeof(double), cudaMemcpyHostToDevice);
            for (k = 0; k < (z / sub_k); k++) {
                // here we can use GPUDirect?
                cudaMemcpy(a_sub_d, (a + i * cross_row + k * cross_col), sub_m * sub_k * sizeof(double), cudaMemcpyHostToDevice);    
                cudaMemcpy(b_sub_d, (b + k * cross_row + j * cross_col), sub_k * sub_n * sizeof(double), cudaMemcpyHostToDevice);
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, sub_k, &alpha, b_sub_d, sub_k, a_sub_d, sub_m, &beta, c_sub_d, sub_m);
            }
            cudaMemcpy((c_h + i * cross_row + j * cross_col), c_sub_d, sub_m * sub_n * sizeof(double), cudaMemcpyDeviceToHost);
        }
    }
    
    cublasDestroy(handle);

    for (int i = 0; i < x * y; ++i) {
        c[i] = (float) c_h[i];
    }

    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(c_sub_d);
    free(c_h);

    return c;
}

float* sequential_blockmm(int x, int y, int z, int sub_m, int sub_n, int sub_k, 
    double *a, double *b, float *c) {
    int i, j, k, ii, jj, kk, i_idx, j_idx, k_idx;
    double alpha = 1.0;
    double beta = 0.0;
    double *a_sub_d, *b_sub_d, *c_sub_d;
    double *a_sub_h, *b_sub_h, *c_sub_h;
    double *c_h;
    cublasHandle_t handle;
    cublasCreate(&handle);

    a_sub_h = (double *) malloc(sizeof(double) * sub_m * sub_k);
    b_sub_h = (double *) malloc(sizeof(double) * sub_k * sub_n);
    c_sub_h = (double *) malloc(sizeof(double) * sub_m * sub_n);
    c_h = (double *) malloc(sizeof(double) * x * y);

    cudaMalloc((void **) &a_sub_d, sizeof(double) * sub_m * sub_k);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * sub_k * sub_n);
    cudaMalloc((void **) &c_sub_d, sizeof(double) * sub_m * sub_n);

    for(i = 0; i < x; i += sub_m) {
        for(j = 0; j < y; j += sub_n) {
            for(k = 0; k < z; k += sub_k) {
                // for (ii = i, i_idx = 0; ii < (i + sub_m); ii++, i_idx++) {
                //     for (jj = j, j_idx = 0; jj < (j + sub_n); jj++, j_idx++) {
                //         c_sub_h[i_idx * sub_n + j_idx] = c_h[ii * y + jj];
                //     }
                // }
                for (ii = i, i_idx = 0; ii < (i + sub_m); ii++, i_idx++) {
                    for (kk = k, k_idx = 0; kk < (k + sub_k); kk++, k_idx++) {
                        a_sub_h[i_idx * sub_n + k_idx] = a[ii*y + kk];         
                    }
                }

                for (jj = j, j_idx = 0; jj < (j + sub_n); jj++, j_idx++) {
                    for (kk = k, k_idx = 0; kk < (k + sub_k); kk++, k_idx++) {
                        b_sub_h[k_idx * sub_n + j_idx] = b[kk * y + jj];
                    }
                }
                cudaMemcpy(a_sub_d, a_sub_h, sub_m * sub_k * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(b_sub_d, b_sub_h, sub_k * sub_n * sizeof(double), cudaMemcpyHostToDevice);
                // cudaMemcpy(c_sub_d, c_sub_h, x * y * sizeof(double), cudaMemcpyHostToDevice);
                // cublasDgemm EXPLANATION ------------------------------------------------
                // the memory layout is different from we know
                // a = [0 1; b = [3 2; 
                //      2 3]      1 0]
                // if use a_d then b_d, c[0][0] will be a[0, 0] * b[0, 0] + a[1, 0] * b[0, 1] = 4
                // with b_d then a_d, c[0][0] will be a[0, 0] * b[0, 0] + a[0, 1] * b[1, 0] = 1
                // maybe that's because inside GPU it uses column major storage.
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_n, sub_k, &alpha, b_sub_d, sub_k, a_sub_d, sub_m, &beta, c_sub_d, sub_m);
                cudaMemcpy(c_sub_h, c_sub_d, sub_m * sub_n * sizeof(double), cudaMemcpyDeviceToHost);

                for (ii = i, i_idx = 0; ii < (i + sub_n); ii++, i_idx++) {
                    for (jj = j, j_idx = 0; jj < (j + sub_n); jj++, j_idx++) {
                        // could be casted to double here?
                        c_h[ii * y + jj] += c_sub_h[i_idx * sub_n + j_idx];
                    }
                }  
            }              
        }
    }  
    
    cublasDestroy(handle);

    for (int i = 0; i < x * y; ++i) {
        c[i] = (float) c_h[i];
    }

    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(c_sub_d);
    free(a_sub_h);
    free(b_sub_h);
    free(c_sub_h);
    free(c_h);

    return c;
}

float* doMultiply2Matrices(int m, int n, int k, const double *a, const double *b, float *c) {
    double alpha = 1.0;
    double beta = 0.0;
    double *a_d, *b_d, *c_d, *c_h;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMalloc((void **) &a_d, sizeof(double) * m * k);
    cudaMalloc((void **) &b_d, sizeof(double) * k * n);
    cudaMalloc((void **) &c_d, sizeof(double) * m * n);
    c_h = (double *) malloc(sizeof(double) * m * n);

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
    
    cudaMemcpy(c_h, c_d, sizeof(double) * m * n, cudaMemcpyDeviceToHost);
    cublasDestroy(handle);

    for (int i = 0; i < m * n; ++i) {
        c[i] = (float) c_h[i];
    }

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(c_h);
    return c;
}

void cpu_verify(double *A, double *B, float *C, unsigned int m, unsigned int n, unsigned int k) {
    const float relativeTolerance = 1e-3;
  
    for(int row = 0; row < m; ++row) {
        for(int col = 0; col < n; ++col) {
            float sum = 0;
            for(unsigned int i = 0; i < k; ++i) {
                // printf("C[%u][%u] = A[%u][%u] * B[%u][%u]\n", row, col, row, i, i, col);
                sum += (float) A[row*k + i]*B[i*n + col];
            }
            float relativeError = (sum - C[row*n + col])/sum;
            if (fabs(relativeError) > relativeTolerance) {
                printf("(%d, %d) = %f, supposed to be %f\n", row, col, C[row*n + col], sum); 
                printf("TEST FAILED\n\n");
                exit(0);
            }
        }
    }
    printf("TEST PASSED\n\n");
}

void gpu_verify(const double *A, const double *B, float *C, unsigned int m, unsigned int n, unsigned int k) {
    const float relativeTolerance = 1e-3;
    float *c_valid = (float *) malloc(sizeof(float) * m * n);
    doMultiply2Matrices(m, n, k, A, B, c_valid);

    for(int row = 0; row < m; ++row) {
        for(int col = 0; col < n; ++col) {
            float relativeError = (c_valid[row*n + col] - C[row*n + col]) / c_valid[row*n + col];
            if (fabs(relativeError) > relativeTolerance) {
                printf("(%d, %d) = %f, supposed to be %f\n", row, col, C[row*n + col], c_valid[row*n + col]); 
                printf("TEST FAILED\n\n");
                exit(0);
            }
        }
    }
    printf("TEST PASSED\n\n");
    free(c_valid);
}


int main(int argc, char** argv) {
    double *a, *b;
    float *c;
    int n, sub_n;
    int a_fd, b_fd;

    if (argc < 3) {
        printf("usage: %s <sequence format path> <tensor format path> <matrix size> <submatrix size>\n", argv[0]);
        exit(1);
    }

    // GEMM configuration.
    a_fd = open(argv[1], O_RDONLY);
    b_fd = open(argv[1], O_RDONLY);

    n = atoi(argv[3]);
    sub_n = atoi(argv[4]);

    a = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, a_fd, 0);
    b = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, b_fd, 0);
  
    // a = (double *) malloc(sizeof(double) * n * n);
    // b = (double *) malloc(sizeof(double) * n * n);

    c = (float *) calloc(n * n, sizeof(float));
    // doMultiply2Matrices(n, n, n, a, b, c);
    sequential_blockmm(n, n, n, sub_n, sub_n, sub_n, a, b, c);


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
    gpu_verify(a, b, c, n, n, n);

    // GEMM configuration.
    int a_tensor_fd = open(argv[2], O_RDONLY);
    int b_tensor_fd = open(argv[2], O_RDONLY);

    double *a_tensor = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, a_tensor_fd, 0);
    double *b_tensor = (double *) mmap(NULL, sizeof(double) * n * n, PROT_READ, MAP_PRIVATE, b_tensor_fd, 0);
    
    memset(c, 0, sizeof(float) * n * n);

    tensor_blockmm(n, n, n, sub_n, sub_n, sub_n, a_tensor, b_tensor, c);

    printf("Reformat from tensor to sequential...\n");
    
    int count = 0;
    float *c_reformat = (float *) calloc(n * n, sizeof(float));
    for (int i = 0; i < n; i += sub_n) {
        for (int j = 0; j < n; j += sub_n) {  
            for(int ii = i; ii < i + sub_n; ii++) {
                for(int jj = j; jj < j + sub_n; jj++) {
                    // printf("ii: %d, jj: %d\n", ii, jj);
                    c_reformat[ii * n + jj] = c[count];
                    count++;
                }
            }
        }
    }
#ifdef DEBUG
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", a_tensor[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("%f ", b_tensor[i * n + j]);
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

    gpu_verify(a, b, c_reformat, n, n, n);
    munmap(a, sizeof(double) * n * n);
    munmap(b, sizeof(double) * n * n);
    close(a_fd);
    close(b_fd);

    munmap(a_tensor, sizeof(double) * n * n);
    munmap(b_tensor, sizeof(double) * n * n);
    close(a_tensor_fd);
    close(b_tensor_fd);

    free(c_reformat);
    free(c);
    return 0;
}