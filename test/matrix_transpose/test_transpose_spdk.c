#include "tensorstore.h"
#include "spdkrpc.h"

// file I/O
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// timing
#include <sys/time.h>
#include <math.h>

int verify(const double *matrix, const double *answer, size_t m) {
    // also need to consider floating point error
    const double relativeTolerance = 1e-3;
    size_t row, col;
    double relativeError;
    for(row = 0; row < m; ++row) {
        for(col = 0; col < m; ++col) {
            if (isnan(matrix[row*m + col])) {
                printf("matrix (%lu, %lu) is NaN\n", row, col);
                return 0; 
            }

            if (isinf(matrix[row*m + col])) {
                printf("matrix (%lu, %lu) is inf\n", row, col);
                return 0; 
            }

            if (isnan(answer[row*m + col])) {
                printf("answer (%lu, %lu) is NaN\n", row, col);
                return 0; 
            }

            if (isinf(answer[row*m + col])) {
                printf("answer (%lu, %lu) is inf\n", row, col);
                return 0; 
            }

            relativeError = (answer[row*m + col] - matrix[row*m + col]) / answer[row*m + col];
            if (fabs(relativeError) > relativeTolerance) {
                printf("(%lu, %lu) = %f, supposed to be %f\n", row, col, matrix[row*m + col], answer[row*m + col]); 
                printf("TEST FAILED\n\n");
                return 0;
            }    
        }
    }
    printf("TEST PASSED\n\n");
    return 1;
}


int transpose_matrix_from_spdk(int id, size_t m, size_t sub_m, double *out_matrix) {
    size_t i, j, ii;
    double *hugepage_addr;
    double *out_ptr;
    struct JSONRPCClient client;
    size_t return_size; 
    int rc;

    struct timeval h_start, h_end;
    unsigned long long fetch_time = 0, transpose_time = 0;
    rc = connect_to_spdkrpc_server(&client);
    if (rc) {
        printf("cannot create conntection to SPDK RPC server");
        return rc;
    }

    hugepage_addr = mmap_to_tensorstore_hugepage();
    for (i = 0; i < m / sub_m; i++) {
        for (j = 0; j < m / sub_m; j++) {
            // memset(hugepage_addr, 0, HUGEPAGE_SZ);
            gettimeofday(&h_start, NULL);
            return_size = tensorstore_request_submatrix(&client, id, j, i);
            gettimeofday(&h_end, NULL);
            fetch_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   

            if (return_size == 0) {
                return -1;
            }
            out_ptr = out_matrix + j * sub_m * m + i * sub_m; 
            // iterate rows in the submatrix
            gettimeofday(&h_start, NULL);
            for (ii = 0; ii < sub_m; ii++) {
                memcpy(out_ptr + ii * m, hugepage_addr + ii * sub_m, sizeof(double) * sub_m);
            }
            gettimeofday(&h_end, NULL);
            transpose_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   
        }   
    }
    munmap(hugepage_addr, HUGEPAGE_SZ);
    printf("data fetch time: %f ms\n", (float) fetch_time / 1000);
    printf("transpose time: %f ms\n", (float) transpose_time / 1000);
    return 0;
 }

int main(int argc, char** argv) {
    int id;
    size_t m, sub_m;
    double *valid_matrix, *seq_matrix;
    int seq_fd;
    double *seq_file;
    struct timeval h_start, h_end;
    unsigned long long duration;

    if (argc < 5) {
        printf("usage: %s <id> <matrix size> <submatrix size> <matrix_path>\n", argv[0]);
        exit(1);
    }

    id = atoi(argv[1]);
    m = atoi(argv[2]);
    sub_m = atoi(argv[3]);

    if (m < sub_m) {
        printf("matrix size has to be larger than submatrix size\n");
        exit(1);
    }

    if (m % sub_m) {
        printf("submatrix size cannot divide matrix size evenly\n");
        exit(1);
    }
    seq_matrix = (double *) malloc(sizeof(double) * m * m);

    gettimeofday(&h_start, NULL);
    transpose_matrix_from_spdk(id, m, sub_m, seq_matrix);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   
    printf("seq_matrix transpose time: %f ms\n", (float) duration / 1000);

    seq_fd = open(argv[4], O_RDONLY, 0644);
    seq_file = (double *) mmap(NULL, sizeof(double) * m * m, PROT_READ, MAP_PRIVATE, seq_fd, 0);
    close(seq_fd);

    valid_matrix = (double *) malloc(sizeof(double) * m * m);

    gettimeofday(&h_start, NULL);
    seq_matrix_transpose(seq_file, valid_matrix, m, m, sub_m, sub_m);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   
    printf("valid_matrix transpose time: %f ms\n", (float) duration / 1000);

    verify(seq_matrix, valid_matrix, m);

    munmap(seq_file, sizeof(double) * m * m);
    free(valid_matrix);
    return 0;
}