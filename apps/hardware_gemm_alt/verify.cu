extern "C" {
    #include "rdma.h"
    #include "timing.h"
    #include "fifo.h"
}

#include "cublasGEMM.h"

#define MAX_THREAD 1024

#define HUGEPAGE_SZ (4UL * 1024UL * 1024UL * 1024UL)
#define M 65536UL
#define SUB_M 1024UL
#define AGGREGATED_SZ (SUB_M * SUB_M * 8UL)

#define IO_QUEUE_SZ (HUGEPAGE_SZ / AGGREGATED_SZ / 2UL)
// #define IO_QUEUE_SZ 1UL

void print_config(struct config_t config);

struct fetch_conf {
    struct resources *res;
    uint64_t m, sub_m;
    double *a_sub_d;
    char *hugepage_addr;
    struct fifo *sending_queue;
    struct fifo *complete_queue;
    struct timing_info *fetch_timing;
    struct timing_info *a_copy_in_timing;
};

struct request_conf {
    struct resources *res;
    uint64_t id;
    uint64_t m, sub_m;
};

struct fifo_entry {
    double *a_sub_d;
};

__global__ void d2h_kernel(const double *din, half *dout, size_t dsize) {
	size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
	if (idx < dsize)
	{
		dout[idx] = din[idx];
	}
}

int verify(const float *C, uint64_t m, uint64_t n) {
    // also need to consider floating point error
    uint64_t row, col;
    int rc = 0;
    for(row = 0; row < m; ++row) {
        for(col = 0; col < n; ++col) {
            if (isnan(C[row*n + col])) {
                printf("(%lu, %lu) is NaN\n", row, col);
                rc = 1; 
                return rc;
            }

            if (isinf(C[row*n + col])) {
                printf("(%lu, %lu) is inf\n", row, col);
                rc = 1; 
                return rc;
            }
        }
    }
    printf("TEST PASSED\n\n");
    return rc;
}

int cudaMemcpyFromMmap(struct fetch_conf *conf, char *dst, const char *src, const size_t length) {
    struct response *res = NULL;

    timing_info_push_start(conf->fetch_timing);
    res = sock_read_offset(conf->res->sock);
    if (res == NULL) {
        fprintf(stderr, "sync error before RDMA ops\n");
        return 1;
    }
    // if (res->id == 0) {
    //     printf("fetching A[%lu][%lu]\n", res->y, res->x);
    // } else {
    //     printf("fetching B[%lu][%lu]\n", res->y, res->x);
    // }

    timing_info_push_end(conf->fetch_timing);

    timing_info_push_start(conf->a_copy_in_timing);
    cudaMemcpy(dst, src + res->offset, length, cudaMemcpyHostToDevice);
    timing_info_push_end(conf->a_copy_in_timing);

    free(res);
    if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
        fprintf(stderr, "sync error before RDMA ops\n");
        return 1;
    }
    // if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
    //     fprintf(stderr, "sync error before RDMA ops\n");
    //     return 1;
    // }
    return 0;
}

void *fetch_thread(void *args) {
    struct fetch_conf *conf = (struct fetch_conf *) args;
    uint64_t i, k;
    uint64_t dsize = conf->sub_m * conf->sub_m;
    double *ptr_a;
    struct fifo_entry *entry = NULL;
    uint64_t count = 0;

    for (k = 0; k < conf->m / conf->sub_m; k++) {
        for (i = 0; i < conf->m / conf->sub_m; i++) {
            entry = (struct fifo_entry *) fifo_pop(conf->complete_queue);
            ptr_a = conf->a_sub_d + dsize * (count % IO_QUEUE_SZ);

            cudaMemcpyFromMmap(conf, (char *) ptr_a, (char *) conf->hugepage_addr, dsize * sizeof(double));
            count++;

            entry->a_sub_d = ptr_a;
            fifo_push(conf->sending_queue, entry);
        }
    }
    return NULL;
}

void *request_thread(void *args) {
    struct request_conf *conf = (struct request_conf *) args;
    uint64_t i, k;
    for (k = 0; k < conf->m / conf->sub_m; k++) {
        for (i = 0; i < conf->m / conf->sub_m; i++) {
            sock_write_request(conf->res->req_sock, conf->id, k, i, SUB_M, 1, 0);
            sock_read_data(conf->res->req_sock);
        }
    }

    sock_write_request(conf->res->req_sock, -1, k, i, SUB_M, 1, 0);
    sock_read_data(conf->res->req_sock);
    return NULL;
}

int spdk_nds_blockSgemm_half_pthread(struct resources *res, uint64_t id, uint64_t m, uint64_t sub_m, float *c) {
    size_t i, j, k;
    size_t cross_row = m * sub_m, cross_col = sub_m * sub_m;
    double *a_sub_d, *b_sub_d, *b;
    half *a_sub_h, *b_sub_h;
    float *c_sub_f;

    float alpha = 1.0;
    float beta = 1.0;
    
    size_t dsize;
    cublasHandle_t handle;

    struct fifo *sending_queue;
    struct fifo *complete_queue; 
    struct fifo_entry *entries = (struct fifo_entry *) calloc(IO_QUEUE_SZ, sizeof(struct fifo_entry));
    struct fifo_entry *entry = NULL;

    struct timing_info *fetch_timing;
    struct timing_info *a_copy_in_timing;
    struct timing_info *b_copy_in_timing;
    struct timing_info *c_copy_in_timing;
    struct timing_info *queue_timing;
    struct timing_info *d2h_timing;
    struct timing_info *gemm_timing;
    struct timing_info *c_copy_out_timing;    
    
    pthread_t f_thread_id; 
    struct fetch_conf f_conf;

    pthread_t r_thread_id; 
    struct request_conf r_conf;

    struct timeval h_start, h_end;
    long duration;

    // initialization
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    dsize = (m / sub_m) * (m / sub_m) * (m / sub_m);
    queue_timing = timing_info_new(dsize);
    if (queue_timing == NULL) {
        printf("cannot create queue_timing\n");
        return -1;
    }

    fetch_timing = timing_info_new(dsize);
    if (fetch_timing == NULL) {
        printf("cannot create fetch_timing\n");
        return -1;
    }

    a_copy_in_timing = timing_info_new(dsize);
    if (a_copy_in_timing == NULL) {
        printf("cannot create a_copy_in_timing\n");
        return -1;
    }

    b_copy_in_timing = timing_info_new(dsize);
    if (b_copy_in_timing == NULL) {
        printf("cannot create b_copy_in_timing\n");
        return -1;
    }

    c_copy_in_timing = timing_info_new(dsize);
    if (c_copy_in_timing == NULL) {
        printf("cannot create c_copy_in_timing\n");
        return -1;
    }

    d2h_timing = timing_info_new(dsize * 2);
    if (d2h_timing == NULL) {
        printf("cannot create d2h_timing\n");
        return -1;
    }

    gemm_timing = timing_info_new(dsize);
    if (gemm_timing == NULL) {
        printf("cannot create gemm_timing\n");
        return -1;
    }

    c_copy_out_timing = timing_info_new(dsize);
    if (c_copy_out_timing == NULL) {
        printf("cannot create c_copy_out_timing\n");
        return -1;
    }

    // it causes problem if size == 1
    sending_queue = fifo_new(IO_QUEUE_SZ * 2);
	if (sending_queue == NULL) {
        printf("cannot create sending_queue\n");
        return -1;
    }
    
    complete_queue = fifo_new(IO_QUEUE_SZ * 2);
	if (complete_queue == NULL) {
        printf("cannot create complete_queue\n");
        return -1;
    }
    
    for (i = 0; i < IO_QUEUE_SZ; i++) {
        fifo_push(complete_queue, entries + i);
    }

    // cuda malloc
    dsize = sub_m * sub_m;
    b = (double *) malloc(sizeof(double) * dsize);

    for (i = 0; i < dsize; i++) {
        b[i] = (double) rand() / RAND_MAX;
    }
    cudaMalloc((void **) &a_sub_d, sizeof(double) * dsize * IO_QUEUE_SZ);
    cudaMalloc((void **) &b_sub_d, sizeof(double) * dsize);
    cudaMalloc((void **) &a_sub_h, sizeof(half) * dsize);
    cudaMalloc((void **) &b_sub_h, sizeof(half) * dsize);
    cudaMalloc((void **) &c_sub_f, sizeof(float) * dsize);

    r_conf.res = res;
    r_conf.id = id;
    r_conf.m = M;
    r_conf.sub_m = SUB_M;
    pthread_create(&r_thread_id, NULL, request_thread, &r_conf); 

    // create thread here
    f_conf.res = res;
    f_conf.m = m;
    f_conf.sub_m = sub_m;
    f_conf.a_sub_d = a_sub_d;
    f_conf.hugepage_addr = res->buf;
    f_conf.sending_queue = sending_queue;
    f_conf.complete_queue = complete_queue;
    f_conf.fetch_timing = fetch_timing;
    f_conf.a_copy_in_timing = a_copy_in_timing;

    timing_info_set_starting_time(queue_timing);
    timing_info_set_starting_time(c_copy_in_timing);
    timing_info_set_starting_time(fetch_timing);
    timing_info_set_starting_time(a_copy_in_timing);
    timing_info_set_starting_time(d2h_timing);
    timing_info_set_starting_time(gemm_timing);
    timing_info_set_starting_time(c_copy_out_timing);
	pthread_create(&f_thread_id, NULL, fetch_thread, &f_conf); 

    gettimeofday(&h_start, NULL);
    // custom block gemm
    for (k = 0; k < m / sub_m; k++) {
        for (i = 0; i < m / sub_m; i++) {
            timing_info_push_start(queue_timing);
            entry = (struct fifo_entry *) fifo_pop(sending_queue);
            timing_info_push_end(queue_timing);

            timing_info_push_start(d2h_timing);
            d2h_kernel<<<(dsize+MAX_THREAD-1)/MAX_THREAD,MAX_THREAD>>>(entry->a_sub_d, a_sub_h, dsize);
            fifo_push(complete_queue, entry);
            timing_info_push_end(d2h_timing);
            for (j = 0; j < m / sub_m; j++) {
                // copy and transform B
                timing_info_push_start(b_copy_in_timing);
                cudaMemcpy(b_sub_d, b, sub_m * sub_m * sizeof(double), cudaMemcpyHostToDevice);
                timing_info_push_end(b_copy_in_timing);

                timing_info_push_start(d2h_timing);
                d2h_kernel<<<(dsize+MAX_THREAD-1)/MAX_THREAD,MAX_THREAD>>>(b_sub_d, b_sub_h, dsize);
                timing_info_push_end(d2h_timing);

                // copy intermediate C back to GPU
                timing_info_push_start(c_copy_in_timing);
                // cudaMemcpy(c_sub_f, (c + i * cross_row + j * cross_col), sub_m * sub_m * sizeof(float), cudaMemcpyHostToDevice);                
                cudaMemcpy(c_sub_f, c, sub_m * sub_m * sizeof(float), cudaMemcpyHostToDevice);                
                timing_info_push_end(c_copy_in_timing);
                
                // gemm
                timing_info_push_start(gemm_timing);
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, sub_m, sub_m, sub_m, &alpha, b_sub_h, CUDA_R_16F, sub_m, a_sub_h, CUDA_R_16F, sub_m, &beta, c_sub_f, CUDA_R_32F, sub_m, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                cudaDeviceSynchronize();
                timing_info_push_end(gemm_timing);

                timing_info_push_start(c_copy_out_timing);
                // cudaMemcpy((c + i * cross_row + j * cross_col), c_sub_f, sub_m * sub_m * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(c, c_sub_f, sub_m * sub_m * sizeof(float), cudaMemcpyDeviceToHost);
                timing_info_push_end(c_copy_out_timing);
            }
        }
    }

    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("GEMM duration: %f ms\n", (float) duration / 1000);    

    printf("Fetch time: %f ms\n", (float) timing_info_duration(fetch_timing) / 1000);
    printf("A Copy in time: %f ms\n", (float) timing_info_duration(a_copy_in_timing) / 1000);
    printf("B Copy in time: %f ms\n", (float) timing_info_duration(b_copy_in_timing) / 1000);
    printf("C copy in time: %f ms\n", (float) timing_info_duration(c_copy_in_timing) / 1000);
    printf("queue waiting time: %f ms\n", (float) timing_info_duration(queue_timing) / 1000);
    printf("d2h time: %f ms\n", (float) timing_info_duration(d2h_timing) / 1000);
    printf("GEMM time: %f ms\n", (float) timing_info_duration(gemm_timing) / 1000);
    printf("C copy out time: %f ms\n", (float) timing_info_duration(c_copy_out_timing) / 1000);

    pthread_join(r_thread_id, NULL); 
    pthread_join(f_thread_id, NULL); 

    struct timestamps *tss = NULL;
    FILE *fptr;
    tss = timing_info_get_timestamps(fetch_timing);
    fptr = fopen("fetch_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(fetch_timing);

    tss = timing_info_get_timestamps(a_copy_in_timing);
    fptr = fopen("a_copy_in_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(a_copy_in_timing);

    tss = timing_info_get_timestamps(b_copy_in_timing);
    fptr = fopen("b_copy_in_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(b_copy_in_timing);

    tss = timing_info_get_timestamps(c_copy_in_timing);
    fptr = fopen("c_copy_in_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(c_copy_in_timing);
    
    tss = timing_info_get_timestamps(queue_timing);
    fptr = fopen("queue_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(queue_timing);

    tss = timing_info_get_timestamps(d2h_timing);
    fptr = fopen("d2h_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(d2h_timing);

    tss = timing_info_get_timestamps(gemm_timing);
    fptr = fopen("gemm_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(gemm_timing);

    tss = timing_info_get_timestamps(c_copy_out_timing);
    fptr = fopen("c_copy_out_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(c_copy_out_timing);

    free(b);
    cublasDestroy(handle);
    cudaFree(a_sub_d);
    cudaFree(b_sub_d);
    cudaFree(a_sub_h);
    cudaFree(b_sub_h);
    cudaFree(c_sub_f);
    fifo_free(sending_queue);
    fifo_free(complete_queue);
    free(entries);    
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
    float *c;
    uint64_t matrix_id, n, sub_n;

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

    if (argc < 5) {
        printf("usage: %s <matrix_id> <n> <sub_n> <port>\n", argv[0]);
        exit(1);
    }
    matrix_id = (uint64_t) atoll(argv[1]);
    n = (uint64_t) atoll(argv[2]);
    sub_n = (uint64_t) atoll(argv[3]);

    config.tcp_port = atoi(argv[4]);
    
    // GEMM configuration
    c = (float *) calloc(n * n, sizeof(float));
    
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
    memset(c, 0, sub_n * sub_n * sizeof(float));

    printf("hugepage starting address is: %p\n", hugepage_addr);

    printf("socket connection\n");
    rc = make_two_tcp_connection(&res, &config);
    if (rc < 0) {
        perror("sock connect");
        exit(1);
    }

    // use the first res to find the device.
    if (resources_find_device(&res, &config)) {
        fprintf(stderr, "failed to find a device for resources\n");
        exit(1);
    }

    if (resources_create(&res, &config)) {
        fprintf(stderr, "failed to create resources\n");
        exit(1);
    }

    /* connect the QPs */
    if (connect_qp(&res, &config)) {
        fprintf(stderr, "failed to connect QPs\n");
        exit(1);
    }

    fprintf(stdout, "running server\n");

    printf("calculating the result of SPDK GEMM\n");
    rc = spdk_nds_blockSgemm_half_pthread(&res, matrix_id, n, sub_n, c);

    // config_destroy(&config);

    munmap(hugepage_addr, BUF_SIZE);
    if (resources_destroy(&res)) {
        fprintf(stderr, "failed to destroy resources\n");
        exit(1);
    }
    close(res.req_sock);

    for (int i = 0; i < 16; i++) {
        printf("%f ", c[i]);
    }
    printf("\n");
    rc = verify(c, sub_n, sub_n);
    free(c);

    printf("test result is %i\n", rc);
    return rc;
}
