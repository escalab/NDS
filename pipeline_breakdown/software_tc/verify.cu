extern "C" {
    #include "rdma.h"
    #include "timing.h"
    #include "fifo.h"
}

#include <fcntl.h>
#include <unistd.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define M 2048UL
#define N M
#define K N
#define P 16UL

#define LDA1 K
#define LDB1 M
#define LDB2 LDB1*N
#define LDC1 N
#define LDC2 LDC1*M

#define SUB_M 512UL
#define THREAD_NUM 8UL

#define HUGEPAGE_SZ (4UL * 1024UL * 1024UL * 1024UL)
#define AGGREGATED_SZ (SUB_M * SUB_M * SUB_M * 8UL)

#define IO_QUEUE_SZ (HUGEPAGE_SZ / AGGREGATED_SZ / 2UL)
// #define IO_QUEUE_SZ 1UL

void print_config(struct config_t config);

struct fetch_conf {
    struct resources *res;
    uint64_t m, sub_m;
    double *d_B;
    char *hugepage_addr;
    struct timing_info *fetch_timing;
    struct timing_info *copy_in_B_timing;
};

__global__ void block_tc_kernel(const double *A, const double *B, double *C) {
    uint64_t m, n, k, p;
    m = blockDim.x * blockIdx.x + threadIdx.x;
    n = blockDim.y * blockIdx.y + threadIdx.y;
    p = blockDim.z * blockIdx.z + threadIdx.z;

    if (m < SUB_M && n < SUB_M && p < P) {
        for (k = 0; k < SUB_M; k++) {      
            C[m + n * SUB_M + p * SUB_M*SUB_M] += B[k + m * SUB_M + n * SUB_M*SUB_M] * A[k + p * K];
        }
    }
}

int cudaMemcpyFromMmap(struct fetch_conf *conf, uint64_t id, uint64_t n, uint64_t m, uint64_t k, uint64_t op, uint64_t which,
    char *dst, const char *src, const size_t length) {
    struct response *res = NULL;

    timing_info_push_start(conf->fetch_timing);
    sock_write_request(conf->res->req_sock, id, n, m, k, op, which);
    sock_read_data(conf->res->req_sock);

    res = sock_read_offset(conf->res->sock);
    if (res == NULL) {
        fprintf(stderr, "sync error before RDMA ops\n");
        return 1;
    }

    timing_info_push_end(conf->fetch_timing);

    timing_info_push_start(conf->copy_in_B_timing);
    cudaMemcpy(dst, src + res->offset, length, cudaMemcpyHostToDevice);
    timing_info_push_end(conf->copy_in_B_timing);

    free(res);
    if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
        fprintf(stderr, "sync error before RDMA ops\n");
        return 1;
    }
    return 0;
}

void generate_data(double *array, uint64_t size) {
    uint64_t i;
    for (i = 0; i < size; i++) {
        array[i] = (((double) rand())/RAND_MAX - 0.5)*100;
    }
}

int nds_tc(struct resources *res, uint64_t id, uint64_t size, uint64_t sub_size, const double *A, double *C) {
    double *d_A;
    double *d_B;
    double *d_C;

    double *sub_B, *sub_C;
    size_t i, n, m, k, p, nn, mm, a, b;

    size_t total_iteration;

    struct timing_info *queue_timing;
    struct timing_info *fetch_timing;
    struct timing_info *copy_in_B_timing;
    struct timing_info *copy_in_C_timing;
    struct timing_info *kernel_timing;
    struct timing_info *copy_out_timing;    
    
    struct fetch_conf f_conf;

    cudaError_t err;

    struct timeval h_start, h_end;
    long duration;

    // initialization
    total_iteration = (M / SUB_M) * (M / SUB_M) * (M / SUB_M);
    queue_timing = timing_info_new(total_iteration);
    if (queue_timing == NULL) {
        printf("cannot create queue_timing\n");
        return -1;
    }

    fetch_timing = timing_info_new(total_iteration);
    if (fetch_timing == NULL) {
        printf("cannot create fetch_timing\n");
        return -1;
    }

    copy_in_B_timing = timing_info_new(total_iteration);
    if (copy_in_B_timing == NULL) {
        printf("cannot create copy_in_B_timing\n");
        return -1;
    }

    copy_in_C_timing = timing_info_new(total_iteration);
    if (copy_in_C_timing == NULL) {
        printf("cannot create copy_in_C_timing\n");
        return -1;
    }

    kernel_timing = timing_info_new(total_iteration);
    if (kernel_timing == NULL) {
        printf("cannot create kernel_timing\n");
        return -1;
    }

    copy_out_timing = timing_info_new(total_iteration);
    if (copy_out_timing == NULL) {
        printf("cannot create copy_out_timing\n");
        return -1;
    }

    sub_B = (double *) malloc(AGGREGATED_SZ);
    sub_C = (double *) malloc(SUB_M * SUB_M * P * sizeof(double));

    cudaMalloc((void **) &d_A, K * P * sizeof(double));
    cudaMalloc((void **) &d_B, AGGREGATED_SZ);
    cudaMalloc((void **) &d_C, SUB_M * SUB_M * P * sizeof(double));

    cudaMemcpy(d_A, A, K * P * sizeof(double), cudaMemcpyHostToDevice);

    // M * N has to be < 1024
    dim3 grid((SUB_M+THREAD_NUM-1)/THREAD_NUM, (SUB_M+THREAD_NUM-1)/THREAD_NUM, (P+THREAD_NUM-1)/THREAD_NUM);
    dim3 block(THREAD_NUM, THREAD_NUM, THREAD_NUM);

    // create thread here
    f_conf.res = res;
    f_conf.m = size;
    f_conf.sub_m = sub_size;
    f_conf.d_B = d_B;
    f_conf.hugepage_addr = res->buf;
    f_conf.fetch_timing = fetch_timing;
    f_conf.copy_in_B_timing = copy_in_B_timing;

    timing_info_set_starting_time(queue_timing);
    timing_info_set_starting_time(fetch_timing);
    timing_info_set_starting_time(copy_in_B_timing);
    timing_info_set_starting_time(kernel_timing);
    timing_info_set_starting_time(copy_out_timing);

    gettimeofday(&h_start, NULL);
    // blockGEMM
    for (n = 0; n < N; n+=SUB_M) {
        for (m = 0; m < M; m+=SUB_M) { 
            timing_info_push_start(copy_in_C_timing);
            cudaMemset(d_C, 0, SUB_M * SUB_M * sizeof(double));
            // for (nn = n, a = 0; nn < n+SUB_M; nn++, a++) {
            //     for (mm = m, b = 0; mm < m+SUB_M; mm++, b++) {
            //         sub_C[b + a * SUB_M] = C[mm + nn * LDC1];
            //     }
            //     cudaMemcpy(d_C, sub_C, SUB_M * SUB_M * sizeof(double), cudaMemcpyHostToDevice);
            // }  
            timing_info_push_end(copy_in_C_timing);

            for (k = 0; k < K; k+=SUB_M) {
                cudaMemcpyFromMmap(&f_conf, id, n/SUB_M, m/SUB_M, k/SUB_M, 1, 0, (char *) d_B, (char *) res->buf, AGGREGATED_SZ);

                timing_info_push_start(kernel_timing);
                block_tc_kernel<<<grid, block>>>(d_A + k, d_B, d_C);
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("CUDA error, error code: %d, error name: %s\n", err, cudaGetErrorString(err));
                }
                timing_info_push_end(kernel_timing);
            }
            // assign C
            timing_info_push_start(copy_out_timing);

            // use cudaMemcpy2D but not the bottleneck.
            cudaMemcpy(sub_C, d_C, SUB_M * SUB_M * P * sizeof(double), cudaMemcpyDeviceToHost);
            // cudaDeviceSynchronize();
            for (p = 0; p < P; p++) {
                for (nn = n, a = 0; nn < n+SUB_M; nn++, a++) {
                    for (mm = m, b = 0; mm < m+SUB_M; mm++, b++) {
                        C[mm + nn * LDC1 + p * LDC2] = sub_C[b + a * SUB_M + p * SUB_M*SUB_M];
                    }
                }
            }
            timing_info_push_end(copy_out_timing);
        }
    }
    sock_write_request(res->req_sock, -1, n, m, k, 1, 0);
    sock_read_data(res->req_sock);
            
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("TTV duration: %f ms\n", (float) duration / 1000);    

    printf("Row fetch time: %f ms\n", (float) timing_info_duration(fetch_timing) / 1000);
    printf("Copy in B time: %f ms\n", (float) timing_info_duration(copy_in_B_timing) / 1000);
    printf("Copy in C time: %f ms\n", (float) timing_info_duration(copy_in_C_timing) / 1000);
    printf("Kernel time: %f ms\n", (float) timing_info_duration(kernel_timing) / 1000);
    printf("copy out time: %f ms\n", (float) timing_info_duration(copy_out_timing) / 1000);

    struct timestamps *tss = NULL;
    FILE *fptr;
    tss = timing_info_get_timestamps(fetch_timing);
    fptr = fopen("fetch_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(fetch_timing);

    tss = timing_info_get_timestamps(copy_in_B_timing);
    fptr = fopen("copy_in_B_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(copy_in_B_timing);
    
    tss = timing_info_get_timestamps(copy_in_C_timing);
    fptr = fopen("copy_in_C_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(copy_in_C_timing);

    tss = timing_info_get_timestamps(kernel_timing);
    fptr = fopen("ttv_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(kernel_timing);

    tss = timing_info_get_timestamps(copy_out_timing);
    fptr = fopen("copy_out_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(copy_out_timing);

    // write the answer here.
    fptr = fopen("answer.bin", "wb");
    fwrite(C, sizeof(double), M * N, fptr);
    fclose(fptr);

    for (i = 0; i < 4; i++) {
        printf("%f ", C[i]);
    }
    printf("\n");
    
    free(sub_B);
    free(sub_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

/******************************************************************************
 * Function: print_config
 *
 * Input
 * none
 *
 * Output
 * none
 *
 * Returns
 * none
 *
 * Description
 * Print out config information
 ******************************************************************************/
void print_config(struct config_t config) {
    fprintf(stdout, " ------------------------------------------------\n");
    fprintf(stdout, " Device name : \"%s\"\n", config.dev_name);
    fprintf(stdout, " IB port : %u\n", config.ib_port);
    if (config.server_name)
        fprintf(stdout, " IP : %s\n", config.server_name);
    fprintf(stdout, " TCP port : %u\n", config.tcp_port);
    if (config.gid_idx >= 0)
        fprintf(stdout, " GID index : %u\n", config.gid_idx);
    fprintf(stdout, " ------------------------------------------------\n\n");
}

int main(int argc, char *argv[]) {
    int rc = 0;
    uint64_t matrix_id, n, sub_n;
    double *A, *C;

    int hugepage_fd;
    char *hugepage_addr;

    // RDMA
    struct resources res;
    struct config_t config = {
        "mlx4_0",  /* dev_name */
        NULL,  /* server_name */
        19875, /* tcp_port */
        1,     /* ib_port */
        0     /* gid_idx */
    };

    // default the iteration is 4 times
    if (argc < 5) {
        printf("usage: %s <matrix_id> <# of vertices> <# of subvertices> <port>\n", argv[0]);
        exit(1);
    } 
    matrix_id = (uint64_t) atoll(argv[1]);
    n = (uint64_t) atoll(argv[2]);
    sub_n = (uint64_t) atoll(argv[3]);
    config.tcp_port = atoi(argv[4]);

    /* print the used parameters for info*/
    print_config(config);
    
    printf("mapping hugepage\n");
    hugepage_fd = open("/dev/hugepages/tensorstore", O_RDWR, 0755);
    if (hugepage_fd < 0) {
        perror("open");
        exit(1);
    }

    hugepage_addr = (char *) mmap(0, BUF_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, hugepage_fd, 0);
    if (hugepage_addr==MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    res.buf = hugepage_addr;
    memset(hugepage_addr, 0, BUF_SIZE);

    printf("hugepage starting address is: %p\n", hugepage_addr);
    printf("socket connection\n");
    rc = make_two_tcp_connection(&res, &config);
    if (rc < 0) {
        perror("sock connect");
        exit(1);
    }

    // generate data
    A = (double *) malloc(K * P * sizeof(double));
    C = (double *) malloc(M * N * P * sizeof(double));

    srand(5);
    generate_data(A, K * P);
    memset(C, 0, M * N * P * sizeof(double));

    printf("calculating the result of pagerank\n");
    rc = nds_tc(&res, matrix_id, n, sub_n, A, C);
    
    close(res.sock);
    close(res.req_sock);
    munmap(hugepage_addr, BUF_SIZE);
    close(hugepage_fd);

    free(A);
    free(C);
    return rc;
}
