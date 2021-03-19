extern "C" {
    #include "timing.h"
}

#include <stdint.h>
#include <sys/time.h>
#include "cublasGEMM.h"

typedef double datatype;
#define gemmtype CUDA_R_64F

void request_submatrix(datatype *seq_matrix, datatype *tensor_matrix, uint64_t x, uint64_t y, uint64_t m, uint64_t sub_m) {
    uint64_t i, offset;

    offset = y * m + x * sub_m;
    for (i = 0; i < sub_m; i++) {
        memcpy(tensor_matrix + i * sub_m, seq_matrix + offset, sizeof(datatype) * sub_m);
        offset += m;
    }
}

int main(int argc, char** argv) {
    uint64_t m, sub_m, i, j, k, dsize;
    datatype *seq_matrix, *tensor_a_matrix, *tensor_b_matrix;
    datatype *a_sub_d, *b_sub_d, *c_sub_d;
    datatype alpha = 1.0;
    datatype beta = 0.0;

    // time
    struct timeval start, end;
    uint64_t duration = 0;
    struct timing_info *a_reshaping_timing;
    struct timing_info *b_reshaping_timing;
    struct timing_info *copy_in_timing;
    struct timing_info *gemm_timing;

    // cuBLAS
    cublasHandle_t handle;

    if (argc < 3) {
        printf("usage: %s <m> <sub_m>\n", argv[0]);
        exit(1);
    }
    
    m = (uint64_t) atoll(argv[1]);
    sub_m = (uint64_t) atoll(argv[2]);

    // initialization
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    dsize = (m / sub_m) * (m / sub_m) * (m / sub_m);
    a_reshaping_timing = timing_info_new(dsize);
    if (a_reshaping_timing == NULL) {
        printf("cannot create a_reshaping_timing\n");
        return -1;
    }

    b_reshaping_timing = timing_info_new(dsize);
    if (b_reshaping_timing == NULL) {
        printf("cannot create b_reshaping_timing\n");
        return -1;
    }

    copy_in_timing = timing_info_new(dsize);
    if (copy_in_timing == NULL) {
        printf("cannot create copy_in_timing\n");
        return -1;
    }

    gemm_timing = timing_info_new(dsize);
    if (gemm_timing == NULL) {
        printf("cannot create gemm_timing\n");
        return -1;
    }

    dsize = sub_m * sub_m;
    seq_matrix = (datatype *) malloc(sizeof(datatype) * m * m);
    tensor_a_matrix = (datatype *) calloc(dsize, sizeof(datatype));
    tensor_b_matrix = (datatype *) calloc(dsize, sizeof(datatype));
    cudaMalloc((void **) &a_sub_d, sizeof(datatype) * dsize);
    cudaMalloc((void **) &b_sub_d, sizeof(datatype) * dsize);
    cudaMalloc((void **) &c_sub_d, sizeof(datatype) * dsize);

    memset(seq_matrix, 128, sizeof(datatype) * m * m);

    gettimeofday(&start, NULL);
    for (i = 0; i < m / sub_m; i++) {
        for (j = 0; j < m / sub_m; j++) {
            for (k = 0; k < m / sub_m; k++) {
                timing_info_push_start(a_reshaping_timing);
                request_submatrix(seq_matrix, tensor_a_matrix, k, i, m, sub_m);
                timing_info_push_end(a_reshaping_timing);

                timing_info_push_start(b_reshaping_timing);
                request_submatrix(seq_matrix, tensor_b_matrix, j, k, m, sub_m);
                timing_info_push_end(b_reshaping_timing);

                timing_info_push_start(copy_in_timing);
                cudaMemcpy(a_sub_d, tensor_a_matrix, dsize * sizeof(datatype), cudaMemcpyHostToDevice);
                cudaMemcpy(b_sub_d, tensor_b_matrix, dsize * sizeof(datatype), cudaMemcpyHostToDevice);
                timing_info_push_end(copy_in_timing);

                timing_info_push_start(gemm_timing);
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_m, sub_m, &alpha, b_sub_d, gemmtype, sub_m, a_sub_d, gemmtype, sub_m, &beta, c_sub_d, gemmtype, sub_m, gemmtype, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                cudaDeviceSynchronize();
                timing_info_push_end(gemm_timing);
            }
        }
    }
    gettimeofday(&end, NULL);
    duration = ((end.tv_sec*1000000 + end.tv_usec) - (start.tv_sec*1000000 + start.tv_usec));
    printf("Reshaping time: %f (msec)\n", (float) duration / 1000.);
    printf("A Reshaping Time: %f ms\n", (float) timing_info_duration(a_reshaping_timing) / 1000);
    printf("B Reshaping Time: %f ms\n", (float) timing_info_duration(b_reshaping_timing) / 1000);
    printf("Copy in time: %f ms\n", (float) timing_info_duration(copy_in_timing) / 1000);
    printf("GEMM time: %f ms\n", (float) timing_info_duration(gemm_timing) / 1000);

    timing_info_free(a_reshaping_timing);
    timing_info_free(b_reshaping_timing);
    timing_info_free(copy_in_timing);
    timing_info_free(gemm_timing);

    cublasDestroy(handle);
    free(seq_matrix);
    free(tensor_a_matrix);
    free(tensor_b_matrix);
    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(c_sub_d);
    return 0;
}