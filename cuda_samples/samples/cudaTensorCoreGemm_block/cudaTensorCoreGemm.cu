/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// CUDA sample demonstrating a GEMM computation using the Warp Matrix Multiply
// and Accumulate API introduced in CUDA 9.

// In this program, the compute_gemm kernel computes the result of a matrix
// multiplication and addition: D = alpha * A * B + beta * C. The dimensions of
// both C and D matrices are M_GLOBAL x N_GLOBAL. The A matrix is M_GLOBAL x
// K_GLOBAL (row-major), the B matrix is K_GLOBAL x N_GLOBAL (column-major). In
// that kernel, each CTA computes one 128 x 128 tile of the resulting matrix per
// iteration. When the tile is computed, the CTA stores it to the global memory
// and begins a new iteration, selecting a new 128 x 128 tile to compute.
// Each CTA consists of eight warps. For the 128 x 128 tile, each warp computes
// eight 16 x 16 subtiles, organized in a 2 x 4 two-dimensional array. Warps
// compute the 16 x 16 subtiles using nvcuda::wmma::mma_sync operations by
// moving through the K_GLOBAL dimension of the A and B matrices and
// accumulating the intermediate result in the local thread state.

// There are a number of simple optimizations used in the algorithm:
// - The CTA copies the 128 x 128 tile of the C matrix from the global memory to
//   shared memory. After that is done, each warp loads the C matrix fragments
//   from shared memory, thus avoiding a random global memory access.
// - On each internal iteration, the CTA copies a portion of the A and B
// matrices from
//   global memory to shared memory. After that, all warps in the CTA reuse the
//   A and B data from shared memory, thus reducing the number of data copies
//   from global memory.
// - The portions of the A and B matrices are stored in shared memory with an
// additional
//   padding (skew) to reduce the number of shared memory access bank conflicts.
//   (See a detailed explanation near the SKEW_HALF macro definition.)
// - When the CTA finishes computing the tiles of the resulting matrix, each
// warp stores
//   its subtiles to shared memory. The CTA then copies the shared memory
//   contents to global memory, again avoiding redundant random global memory
//   accesses.
// - Note that the CTA tile size is chosen to maximize the GPU register
// utilization,
//   but carefully enough to avoid local memory use.

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

// Externally configurable parameters.

#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 0
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 1
#endif

// GPU configuration.
#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 32
#define N 32
#define K 32

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 half-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 4
#else
#define CHUNK_K 8
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B
// matrix in shared memory to minimize possible bank conflicts. Before
// performing the nvcuda::wmma::mma_sync operation, the warp must load the
// matrix data using the nvcuda::wmma::load_matrix_sync operation. Although the
// memory access pattern is not specified for that function, each lane in the
// warp can read one or multiple matrix elements from different matrix rows or
// columns. For shared memory, such access can result in bank conflicts if
// different rows / columns of the matrix map to the same bank. By shifting each
// row and column by a few bytes, we make sure that they map to different banks,
// thus reducing the number of possible bank conflicts. The number of 8 two-byte
// "half" elements is chosen as the minimum possible shift because we must keep
// each row and column 128-bit aligned, as required by
// nvcuda::wmma::load_matrix_sync.
#define SKEW_HALF 8

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

using namespace nvcuda;

__host__ void init_host_matrices(half *a, half *b, float *c, int M_GLOBAL, int N_GLOBAL, int K_GLOBAL) {
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      a[i * K_GLOBAL + j] = (half)(rand() % 3);
    }
  }

  for (int i = 0; i < N_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      b[i * K_GLOBAL + j] = (half)(rand() % 3);
    }
  }

  for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
    c[t] = static_cast<float>(rand() % 3);
  }
}

__host__ void init_host_matrices(half *a, int M_GLOBAL, int N_GLOBAL) {
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < N_GLOBAL; j++) {
      a[i * N_GLOBAL + j] = (half) (rand() % 3);
    }
  }
}

__host__ void init_host_matrices(double *a, int M_GLOBAL, int N_GLOBAL) {
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < N_GLOBAL; j++) {
      a[i * N_GLOBAL + j] = ((double) rand() / RAND_MAX);
    }
  }
}

__host__ void init_host_matrices(float *c, int M_GLOBAL, int N_GLOBAL) {
  for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
    c[t] = static_cast<float>(rand() % 3);
  }
}


__global__ void half_conversion_kernel(double *din, half *dout, int dsize) {
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
	if (idx < dsize)
	{
		dout[idx] = din[idx];
	}
}

// Calculate AB with NVIDIA TensorCores
// Kernel executed by 1 Warp (32 Threads)
__global__ void tensorOp(half *a, half *b, float *c) {
  // Tile using a 2D grid
  // int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // [0, 1]
  // int warpN = (blockIdx.y * blockDim.y + threadIdx.y); // [0, 1]
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
  int cRow = warpM * WMMA_M;
  int cCol = warpN * WMMA_N;
  // printf("warpM: %d, warpN: %d\n", warpM, warpN);
  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  nvcuda::wmma::load_matrix_sync(c_frag, c + cRow * N + cCol, N, wmma::mem_row_major);

  // Loop over k
  for (int i = 0; i < K; i += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = i;
    int bCol = warpN * WMMA_N;
    int bRow = i;

    // Bounds checking
    if (aRow < M && aCol < K && bRow < K && bCol < N) {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, a + aCol + aRow * M, M);
      wmma::load_matrix_sync(b_frag, b + bCol + bRow * K, K);

      // 4. Perform the matrix multiplication
      nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
  }
  // 5. Store the result from fragment to global
  nvcuda::wmma::store_matrix_sync(c + cRow * N + cCol, c_frag, N, nvcuda::wmma::mem_row_major);
}

// too complicated and use too many macros to ignore this first
/**
__global__ void compute_gemm(const half *A, const half *B, const float *C,
                             float *D, float alpha, float beta) {
  extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  // Offset in shared memory from which the B matrix is stored.
  const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

  // This pointer is used to access the C and D matrix tiles this warp computes.
  float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
                               (warpId / 2) * SHMEM_STRIDE * K * 2 +
                               (warpId % 2) * SHMEM_OFFSET;

  // This pointer is used to stream the C and D matrices block-wide tile to and
  // from shared memory.
  float *shmem_warp_stream_ptr =
      (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may
  // result in a loss of precision). Zero still needs to be specially handled
  // though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the
  // matrix to the right and down, and selects the next tile to compute. Once
  // there's no such tile, all warps in this CTA exit.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_tile_i =
        ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
    const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) {
      break;
    }

    // This warp's pointer to the C matrix data to copy memory from to shared
    // memory.
    const size_t gmem_idx =
        (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
    const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

    // Stream multiple C tiles to shared memory.
#pragma unroll
    for (int i = 0; i < K; i++) {
      typedef int4 copy_t;

      *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
          *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
            laneId);
    }

    __syncthreads();

    // These fragments will accumulate the result of A and B matrix fragment
    // multiplications along the K_GLOBAL dimension.
    wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES]
                                                       [WARP_ROW_TILES];

    // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        const float *tile_ptr =
            shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // Scale the C matrix.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        for (int t = 0; t < c[i][j].num_elements; t++) {
          c[i][j].x[t] *= beta;
        }
      }
    }

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
    const half *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] +
                                           M * K_GLOBAL * (warpId % 4) * 2)
                                        : (&B[block_tile_j * N * K_GLOBAL] +
                                           N * K_GLOBAL * (warpId % 4) * 2);

    // Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest copy
      // the B matrix.
      size_t shmem_idx =
          warpId < (WARPS_PER_BLOCK / 2)
              ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
              : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K +
                                (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
                       (laneId % CHUNK_COPY_LINE_LANES);

      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
      for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
           i++) {
        // Copy 16 bytes at once in each lane.
        *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
            *lane_ptr;

        // Advance the global memory pointer and the shared memory index.
        lane_ptr =
            (int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
        shmem_idx += CHUNK_COPY_LINES_PER_WARP;
      }

      __syncthreads();

      // Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major>
            a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major>
            b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
          const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

          wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b = shmem_idx_b_off +
                                   (WARP_ROW_TILES * N) * (warpId % 2) +
                                   (j * N);
              const half *tile_ptr = &shmem[shmem_idx_b][k_step * K];

              wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
            }

            wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
          }
        }
      }

      __syncthreads();
    }

      // Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        // Uniform, point-wise transformations of ALL fragment elements by ALL
        // threads in the warp are well-defined even though element indices
        // within fragment storage are not defined.
        for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;

        float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // Now that shared memory contains all the D tiles, stream them to global
    // memory.
    float *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
    for (int i = 0; i < K; i++) {
      *((int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
          *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
    }

    __syncthreads();
  }
}
*/

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is a less performant version of the compute_gemm kernel. It is
// designed for
//       demonstration purposes only to show the CUDA WMMA API use without
//       relying on availability of the shared memory.
__global__ void simple_wmma_gemm(half *a, half *b, float *c, float *d, int m_ld,
                                 int n_ld, int k_ld, float alpha, float beta) {
  // Leading dimensions. Packed with no transpositions.
  int lda = m_ld;
  int ldb = k_ld;
  int ldc = n_ld;

  // Tile using a 2D grid
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  // Loop over k
  for (int i = 0; i < k_ld; i += WMMA_K) {
    int aCol = i;
    int aRow = warpM * WMMA_M;

    int bCol = i;
    int bRow = warpN * WMMA_N;

    // Bounds checking
    if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
      wmma::load_matrix_sync(b_frag, b + bCol + bRow * ldb, ldb);

      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Load in the current value of c, scale it by beta, and add this our result
  // scaled by alpha
  int cCol = warpN * WMMA_N;
  int cRow = warpM * WMMA_M;

  if (cRow < m_ld && cCol < n_ld) {
    wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc,
                           wmma::mem_row_major);

    for (int i = 0; i < c_frag.num_elements; i++) {
      c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }

    // Store the output
    wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc,
                            wmma::mem_row_major);
  }
}

__host__ void matMultiplyOnHost(half *A, half *B, float *C, float alpha,
                                float beta, int numARows, int numAColumns,
                                int numBRows, int numBColumns, int numCRows,
                                int numCColumns) {
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      float temp = 0.0;

      for (int k = 0; k < numAColumns; k++) {
        temp += (float)A[i * numAColumns + k] * (float)B[j * numBRows + k];
      }

      C[i * numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
    }
  }
}

__host__ void matMultiplyOnHost(double *A, double *B, float *C, float alpha,
                                float beta, int numARows, int numAColumns,
                                int numBRows, int numBColumns, int numCRows,
                                int numCColumns) {
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      for (int k = 0; k < numAColumns; k++) {
        // beware of row-major or col-major store
        // C[i * numCColumns + j] += (float)A[i * numAColumns + k] * (float)B[j * numBRows + k];
        C[i * numCColumns + j] += (float)A[i * numAColumns + k] * (float)B[k * numBColumns + j];
      }

      // C[i * numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
    }
  }
}

__host__ void blockMatMultiplyOnHost(double *A, double *B, float *C, float alpha,
  float beta, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
  int i, j, k, ii, jj, kk;
  for (i = 0; i < numCRows; i += M) {
    for (j = 0; j < numCColumns; j += N) {
      for (k = 0; k < numAColumns; k += K) {
        for (ii = i; ii < (i + M); ii++) {
          for (jj = j; jj < (j + N); jj++) {
            for (kk = k; kk < (k + K); kk++) {
              C[ii * numCColumns + jj] += A[ii * numAColumns + kk] * B[kk * numBColumns + jj];
            }
          }
        }
      }
      // C[i * numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
    }
  }
}


int main(int argc, char **argv) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  float milliseconds = 0;
  int M_TILES, N_TILES, K_TILES, M_GLOBAL, N_GLOBAL, K_GLOBAL;
  cudaEvent_t start, stop;
  FILE *fp;

  int i, j, k;
  size_t dsize = M * K;
  dim3 gridDim;
  dim3 blockDim;
  printf("Initializing...\n");


#if CPU_DEBUG
  if (argc < 6) {
    printf("usage: %s <A_matrix_path> <m_global> <n_global> <k_global> <answer_path>\n", argv[0]);
    exit(1);
  }
#else
  if (argc < 5) {
    printf("usage: %s <A_matrix_path> <m_global> <n_global> <k_global>\n", argv[0]);
    exit(1);
  }
#endif

  // GEMM configuration.
  fp = fopen(argv[1], "rb");
  M_GLOBAL = atoi(argv[2]);
  N_GLOBAL = atoi(argv[3]);
  K_GLOBAL = atoi(argv[4]);

  M_TILES = M_GLOBAL / M / WMMA_M;
  N_TILES = N_GLOBAL / N / WMMA_N;
  K_TILES = K_GLOBAL / K / WMMA_K;
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
  
  printf("The warp size is %d.\n", deviceProp.warpSize);

  // blockDim.x must be a multple of warpSize
  // 128x4 means we have 16 warps and a block computes a 64x64 output tile
  // for RTX 2080, we have 1024 threads per block.
  blockDim.y = N / WMMA_N; // 32 / 16 = 2
  blockDim.x = deviceProp.warpSize * blockDim.y; // 32 * 2 = 64
  gridDim.x = (M_GLOBAL + (M * blockDim.x / deviceProp.warpSize - 1)) /
              (M * blockDim.x / deviceProp.warpSize);
  gridDim.y = (N_GLOBAL + N * blockDim.y - 1) / (N * blockDim.y);
  printf("gridDim.x=%d, gridDim.y=%d, blockDim.x=%d, blockDim.y=%d\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  printf("M: %d (%d x %d x %d)\n", M_GLOBAL, M, WMMA_M, M_TILES);
  printf("N: %d (%d x %d x %d)\n", N_GLOBAL, N, WMMA_N, N_TILES);
  printf("K: %d (%d x %d x %d)\n", K_GLOBAL, K, WMMA_K, K_TILES);

  double *A_h = NULL;
  double *B_h = NULL;
  float *C_h = NULL;
  A_h = (double *)malloc(sizeof(double) * M_GLOBAL * K_GLOBAL);
  B_h = (double *)malloc(sizeof(double) * K_GLOBAL * N_GLOBAL);
  C_h = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

#if CPU_DEBUG
  double *A_submatrix_h = NULL;
  double *B_submatrix_h = NULL;
  float *C_submatrix_h = NULL;
  float *result_host = NULL;
  float *answer = NULL;

  A_submatrix_h = (double *)malloc(sizeof(double) * M * K);
  B_submatrix_h = (double *)malloc(sizeof(double) * K * N);
  C_submatrix_h = (float *)malloc(sizeof(float) * M * N);
  answer = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
  result_host = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
  memset(result_host, 0, sizeof(float) * M_GLOBAL * N_GLOBAL);
#endif

  int count;
  count = fread(A_h, sizeof(double), M_GLOBAL * K_GLOBAL, fp);
  if (count != M_GLOBAL * K_GLOBAL) {
    printf("read num of element mismatched! count: %d, matrix_size: %d\n",count, M_GLOBAL * K_GLOBAL);
  }

  fseek(fp, 0, SEEK_SET);

  count = fread(B_h, sizeof(double), K_GLOBAL * N_GLOBAL, fp);
  if (count != K_GLOBAL * N_GLOBAL) {
    printf("read num of element mismatched! count: %d, matrix_size: %d\n",count, K_GLOBAL * N_GLOBAL);
  }
  
  // init_host_matrices(B_h, N_GLOBAL, K_GLOBAL);
  // init_host_matrices(C_h, M_GLOBAL, N_GLOBAL);
  memset(C_h, 0, sizeof(float) * M_GLOBAL * N_GLOBAL);

  // printf("A = \n");
  // for (i = 0; i < M_GLOBAL; i++) {
  //   for (j = 0; j < K_GLOBAL; j++) {
  //     printf("%f ", A_h[i*K_GLOBAL+j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  // printf("B = \n");
  // for (i = 0; i < K_GLOBAL; i++) {
  //   for (j = 0; j < N_GLOBAL; j++) {
  //     printf("%f ", B_h[i*N_GLOBAL+j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");


  // printf("C = \n");
  // for (i = 0; i < M_GLOBAL; i++) {
  //   for (j = 0; j < N_GLOBAL; j++) {
  //     printf("%f ", C_h[i*N_GLOBAL+j]);
  //   }
  //   printf("\n");
  // }

  printf("\n");

  double *A_double = NULL;
  half *A = NULL;
  double *B_double = NULL;
  half *B = NULL;
  float *C = NULL;
  float *D = NULL;

  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&A_double),
                             sizeof(double) * M * K));  
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&A),
                             sizeof(half) * M * K));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&B_double),
                             sizeof(double) * M * K));  
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


  printf("Preparing data for GPU...\n");
  checkCudaErrors(cudaMemset(C, 0, sizeof(float) * M * N));

  checkCudaErrors(cudaEventRecord(start));
  // custom block gemm
  int cross_row = M_GLOBAL * K, cross_col = M * K;
  for (i = 0; i < (M_GLOBAL / M); i++) {
    for (j = 0; j < (N_GLOBAL / N); j++) {
        checkCudaErrors(cudaMemcpy(C, (C_h + i * cross_row + j * cross_col), M * N * sizeof(float), cudaMemcpyHostToDevice));
        for (k = 0; k < (K_GLOBAL / K); k++) {
            // fill the block
            // printf("%p\n", A_h + i * cross_row + k * cross_col);
            checkCudaErrors(cudaMemcpy(B_double, (B_h + k * cross_row + j * cross_col), K * N * sizeof(double), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(A_double, (A_h + i * cross_row + k * cross_col), M * K * sizeof(double), cudaMemcpyHostToDevice));    
            half_conversion_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(A_double, A, dsize);
            half_conversion_kernel<<<(dsize+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(B_double, B, dsize);
            tensorOp<<<1, blockDim>>>(A, B, C);
        }
        checkCudaErrors(cudaMemcpy((C_h + i * cross_row + j * cross_col), C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    }
  }

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Time: %f ms\n", milliseconds);
  printf("TFLOPS: %.2f\n", static_cast<double>((static_cast<double>(M_GLOBAL) *
                                                N_GLOBAL * K_GLOBAL * 2) /
                                               (milliseconds / 1000.)) / 1e12);
#if CPU_DEBUG
  printf("Verifying correctness of the computations...\n");
  // matMultiplyOnHost(A_h, B_h, result_host, alpha, beta, M_GLOBAL, K_GLOBAL,
  //                   K_GLOBAL, N_GLOBAL, M_GLOBAL, N_GLOBAL);
  checkCudaErrors(cudaEventRecord(start));
  // blockMatMultiplyOnHost(A_h, B_h, result_host, alpha, beta, M_GLOBAL, K_GLOBAL,
  //   K_GLOBAL, N_GLOBAL, M_GLOBAL, N_GLOBAL);
  
  // custom block gemm
  cross_row = M_GLOBAL * K;
  cross_col = K * K;

  for (i = 0; i < (M_GLOBAL / K); i++) {
    for (j = 0; j < (N_GLOBAL / N); j++) {
        memcpy(C_submatrix_h, (result_host + i * cross_row + j * cross_col), K * N * sizeof(float));
        for (k = 0; k < (K_GLOBAL / K); k++) {
            // fill the block
            memcpy(A_submatrix_h, (A_h + i * cross_row + k * cross_col), K * K * sizeof(double));
            memcpy(B_submatrix_h, (B_h + k * cross_row + j * cross_col), K * N * sizeof(double));
            matMultiplyOnHost(A_submatrix_h, B_submatrix_h, C_submatrix_h, alpha, beta, K, K, K, N, K, N);
        }
        memcpy((result_host + i * cross_row + j * cross_col), C_submatrix_h, K * N * sizeof(float));
    }
  }
  
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  printf("Time: %f ms\n", milliseconds);

  fp = fopen(argv[5], "rb");
  fread(answer, sizeof(float), M_GLOBAL * N_GLOBAL, fp);
  fclose(fp);

  count = 0;
  for (int i = 0; i < M_GLOBAL * N_GLOBAL; i++) {
    if (fabs(answer[i] - result_host[i]) > 0.1f) {
      printf("mismatch i=%d answer=%f result_host=%f\n", i, answer[i], result_host[i]);
      count++;
    }
  }

  if (count == 0) {
    printf("host computation test passed\n");
  }

  count = 0;
  for (int i = 0; i < M_GLOBAL * N_GLOBAL; i++) {
    if (fabs(C_h[i] - result_host[i]) > 0.1f) {
      printf("mismatch i=%d C_h=%f result_host=%f\n", i, C_h[i], result_host[i]);
      count++;
    }             
  }

  if (count == 0) {
    printf("TensorOP test passed\n");
  }

  free(answer);

  free(result_host);
  free(A_submatrix_h);
  free(B_submatrix_h);
  free(C_submatrix_h);
#endif

  free(A_h);
  free(B_h);
  free(C_h);
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(A_double)));
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(A)));
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(B_double)));
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(B)));
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(C)));
  checkCudaErrors(cudaFree(reinterpret_cast<void *>(D)));
  return 0;
}
