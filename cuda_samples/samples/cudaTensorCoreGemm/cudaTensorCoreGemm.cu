#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#define M 16
#define N 16
#define K 16

#define CPU_DEBUG 1


__host__ void matMultiplyOnHost(half *A, half *B, float *C, int numARows, int numAColumns,
  int numBRows, int numBColumns, int numCRows, int numCColumns) {
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      float temp = 0.0;

      for (int k = 0; k < numAColumns; k++) {
        temp += (float)A[i * numAColumns + k] * (float)B[j * numBRows + k];
      }

      C[i * numCColumns + j] = temp + C[i * numCColumns + j];
    }
  }
}


// Calculate AB with NVIDIA TensorCores
// Kernel executed by 1 Warp (32 Threads)
__global__ void tensorOp(float *D, half *A, half *B) {
    // 1.Declare the fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, half, nvcuda::wmma::col_major> Amat;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, half, nvcuda::wmma::col_major> Bmat;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, float, void> Cmat;
    
    // 2.Initialize the output to zero
    nvcuda::wmma::fill_fragment(Cmat, 0.0f);

    //3.Load the inputsintothefragments
    nvcuda::wmma::load_matrix_sync(Amat, A, M);
    nvcuda::wmma::load_matrix_sync(Bmat, B, K);

    //4.Perform the matrix multiplication
    nvcuda::wmma::mma_sync(Cmat, Amat, Bmat, Cmat);

    //5.Store the result from fragment to global
    nvcuda::wmma::store_matrix_sync(D, Cmat, M, nvcuda::wmma::mem_col_major);
}

__host__ void init_host_matrices(half *a, half *b, float *c) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < K; j++) {
        a[i * K + j] = (half)(rand() % 3);
      }
    }
  
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < K; j++) {
        b[i * K + j] = (half)(rand() % 3);
      }
    }
  }

int main(int argc, char **argv) {
    printf("Initializing...\n");
  
    int dev = findCudaDevice(argc, (const char **)argv);
  
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
  
    // Tensor cores require a GPU of Volta (SM7X) architecture or higher.
    if (deviceProp.major < 7) {
      printf(
          "cudaTensorCoreGemm requires SM 7.0 or higher to use Tensor "
          "Cores.  Exiting...\n");
      exit(EXIT_WAIVED);
    }
  
    printf("M: %d \n", M);
    printf("N: %d \n", N);
    printf("K: %d \n", K);
  
    half *A_h = NULL;
    half *B_h = NULL;
    float *C_h = NULL;
  #if CPU_DEBUG
    float *result_hD = NULL;
    float *result_host = NULL;
  #endif
  
    A_h = (half *)malloc(sizeof(half) * M * K);
    B_h = (half *)malloc(sizeof(half) * K * N);
    C_h = (float *)malloc(sizeof(float) * M * N);
  #if CPU_DEBUG
    result_hD = (float *)malloc(sizeof(float) * M * N);
    result_host = (float *)malloc(sizeof(float) * M * N);
  #endif
  
    half *A = NULL;
    half *B = NULL;
    float *C = NULL;
    float *D = NULL;
  
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&A),
                               sizeof(half) * M * K));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&B),
                               sizeof(half) * N * K));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&C),
                               sizeof(float) * M * N));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&D),
                               sizeof(float) * M * N));
  
    assert(((unsigned long long)A) % 128 == 0);
    assert(((unsigned long long)B) % 128 == 0);
    assert(((unsigned long long)C) % 128 == 0);
    assert(((unsigned long long)D) % 128 == 0);
  
    init_host_matrices(A_h, B_h, C_h);
  
    printf("Preparing data for GPU...\n");
  
    checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half) * M * K,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half) * N * K,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(C, C_h, sizeof(float) * M * N,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(D, 0, sizeof(float) * M * N));
  
    cudaEvent_t start, stop;
  
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));
  

    printf("Computing... using simple_wmma_gemm kernel\n");
    tensorOp<<<1, 32>>>(D, A, B);
#if CPU_DEBUG
    checkCudaErrors(cudaMemcpy(result_hD, D,
                                sizeof(float) * M * N,
                                cudaMemcpyDeviceToHost));
#endif
  
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
  
  #if CPU_DEBUG
    printf("Verifying correctness of the computations...\n");
  
    memcpy(result_host, C_h, sizeof(float) * M * N);
  
    matMultiplyOnHost(A_h, B_h, result_host, M, K,
                      K, N, M, N);
  
    for (int i = 0; i < N * M; i++) {
      if (fabs(result_hD[i] - result_host[i]) > 0.1f)
        printf("mismatch i=%d result_hD=%f result_host=%f\n", i, result_hD[i],
               result_host[i]);
    }
    free(result_hD);
    free(result_host);
  #endif
  
    float milliseconds = 0;
  
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
  
    printf("Time: %f ms\n", milliseconds);
    printf("TFLOPS: %.2f\n", static_cast<double>((static_cast<double>(M) *
                                                  N * K * 2) /
                                                 (milliseconds / 1000.)) /
                                 1e12);
  
    free(A_h);
    free(B_h);
    free(C_h);
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(A)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(B)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(C)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(D)));
  
    return 0;
  }