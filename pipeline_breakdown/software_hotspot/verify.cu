extern "C" {
    #include "rdma.h"
    #include "timing.h"
    #include "fifo.h"
}
// CUDA runtime
#include <cuda_runtime.h>

#include "hotspot.h"

#define HUGEPAGE_SZ (4UL * 1024UL * 1024UL * 1024UL)
#define M 65536UL
#define SUB_M 1024UL
#define AGGREGATED_SZ (SUB_M * SUB_M * 8UL)

#define IO_QUEUE_SZ (HUGEPAGE_SZ / AGGREGATED_SZ)
// #define IO_QUEUE_SZ 1UL

void print_config(struct config_t config);

struct fetch_conf {
    struct resources *res;
    uint64_t m, sub_m;
    uint64_t total_iterations, num_iterations;
    double *a_sub_d, *b_sub_d;
    char *hugepage_addr;
    struct fifo *sending_queue;
    struct fifo *complete_queue;
    struct timing_info *fetch_timing;    
};

struct request_conf {
    struct resources *res;
    uint64_t id;
    uint64_t m, sub_m;
    uint64_t total_iterations, num_iterations;
};

struct fifo_entry {
    struct response *temp_resp;
    struct response *power_resp;
};

void *fetch_thread(void *args) {
    struct fetch_conf *conf = (struct fetch_conf *) args;
    uint64_t i, j, k, t;
    struct fifo_entry *entry = NULL;

    for (t = 0; t < conf->total_iterations; t+=conf->num_iterations) {
        for (i = 0; i < conf->m / conf->sub_m; i++) {
            for (j = 0; j < conf->m / conf->sub_m; j++) {
                entry = (struct fifo_entry *) fifo_pop(conf->complete_queue);
    
                timing_info_push_start(conf->fetch_timing);
                entry->temp_resp = sock_read_offset(conf->res->sock);
                if (entry->temp_resp == NULL) {
                    fprintf(stderr, "sync error before RDMA ops\n");
                    return NULL;
                }
                if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
                    fprintf(stderr, "sync error before RDMA ops\n");
                    return NULL;
                }

                entry->power_resp = sock_read_offset(conf->res->sock);
                if (entry->power_resp == NULL) {
                    fprintf(stderr, "sync error before RDMA ops\n");
                    return NULL;
                }
    
                if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
                    fprintf(stderr, "sync error before RDMA ops\n");
                    return NULL;
                }

                timing_info_push_end(conf->fetch_timing);                
                fifo_push(conf->sending_queue, entry);
            }
        }    
    }
    return NULL;
}

void *request_thread(void *args) {
    struct request_conf *conf = (struct request_conf *) args;
    uint64_t i, j, t;
    for (t = 0; t < conf->total_iterations; t+=conf->num_iterations) {
        for (i = 0; i < conf->m / conf->sub_m; i++) {
            for (j = 0; j < conf->m / conf->sub_m; j++) {
                sock_write_request(conf->res->req_sock, conf->id, j, i, SUB_M, 1, 0);
                sock_read_data(conf->res->req_sock);
                sock_write_request(conf->res->req_sock, conf->id, j, i, SUB_M, 1, 1);
                sock_read_data(conf->res->req_sock);
            }
        }
    }
    sock_write_request(conf->res->req_sock, -1, j, i, SUB_M, 1, 0);
    sock_read_data(conf->res->req_sock);
    return NULL;
}

__host__ int spdk_nds_hotspot(struct resources *res, uint64_t id, uint64_t m, uint64_t sub_m) {
    size_t i, j;

    size_t dsize;
    int grid_rows = sub_m, grid_cols = sub_m;

    double *MatrixOut; 
    double *MatrixTempIn, *MatrixTempOut, *MatrixPower;

    double t_chip = 0.0005;
    double chip_height = 0.016;
    double chip_width = 0.016;

    // Rodinia version does more than 1 iteration but here just using 1 iteration can also demonstrate NDS.
    uint64_t t, total_iterations = 1, num_iterations = 1;

    int row = sub_m, col = sub_m;

    struct fifo *sending_queue;
    struct fifo *complete_queue; 
    struct fifo_entry *entries = (struct fifo_entry *) calloc(IO_QUEUE_SZ, sizeof(struct fifo_entry));
    struct fifo_entry *entry = NULL;

    struct timing_info *fetch_timing;
    struct timing_info *copy_in_timing;
    struct timing_info *memset_timing;
    struct timing_info *kernel_timing;
    struct timing_info *copy_out_timing;    

    struct timeval h_start, h_end;
    long duration;

    cudaError_t err;

    // initialization
    dsize = (m / sub_m) * (m / sub_m);

    fetch_timing = timing_info_new(dsize);
    if (fetch_timing == NULL) {
        printf("cannot create fetch_timing\n");
        return -1;
    }

    copy_in_timing = timing_info_new(dsize);
    if (copy_in_timing == NULL) {
        printf("cannot create copy_in_timing\n");
        return -1;
    }

    memset_timing = timing_info_new(dsize);
    if (memset_timing == NULL) {
        printf("cannot create memset_timing\n");
        return -1;
    }

    kernel_timing = timing_info_new(dsize);
    if (kernel_timing == NULL) {
        printf("cannot create kernel_timing\n");
        return -1;
    }

    copy_out_timing = timing_info_new(dsize);
    if (copy_out_timing == NULL) {
        printf("cannot create copy_out_timing\n");
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

    int pyramid_height = 1; // number of iterations

    # define EXPAND_RATE 2 // add one iteration will extend the pyramid base by 2 per each borderline
    int borderCols = (pyramid_height)*EXPAND_RATE/2;
    int borderRows = (pyramid_height)*EXPAND_RATE/2;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

    // cuda malloc
    dsize = sub_m * sub_m;

    MatrixOut = (double *) calloc(dsize, sizeof(double));
    cudaMalloc((void**) &MatrixTempIn, sizeof(double) * dsize);
    cudaMalloc((void**) &MatrixTempOut, sizeof(double) * dsize);
    cudaMalloc((void**) &MatrixPower, sizeof(double) * dsize);

    printf("pyramidHeight: %d\ngridSize: [%d, %d]\nborder:[%d, %d]\nblockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n",\
	pyramid_height, grid_cols, grid_rows, borderCols, borderRows, blockCols, blockRows, smallBlockCol, smallBlockRow);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(blockCols, blockRows);  
	
	double grid_height = chip_height / grid_rows;
	double grid_width = chip_width / grid_cols;

	double Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	double Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	double Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	double Rz = t_chip / (K_SI * grid_height * grid_width);

	double max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	double step = PRECISION / max_slope;
    double time_elapsed = 0.001;

    timing_info_set_starting_time(memset_timing);
    timing_info_set_starting_time(fetch_timing);
    timing_info_set_starting_time(copy_in_timing);
    timing_info_set_starting_time(kernel_timing);
    timing_info_set_starting_time(copy_out_timing);

    printf("Start computing the transient temperature\n");
    gettimeofday(&h_start, NULL);
    
    for (t = 0; t < total_iterations; t+=num_iterations) {
        for (i = 0; i < m / sub_m; i++) {
            for (j = 0; j < m / sub_m; j++) {
                timing_info_push_start(memset_timing);
                cudaMemset(MatrixTempOut, 0, dsize * sizeof(double));
                timing_info_push_end(memset_timing);

                timing_info_push_start(fetch_timing);
                // fetching MatrixTempIn
                entry = (struct fifo_entry *) fifo_pop(complete_queue);
                sock_write_request(res->req_sock, id, j, i, SUB_M, 1, 0);
                sock_read_data(res->req_sock);
                entry->temp_resp = sock_read_offset(res->sock);
                if (entry->temp_resp == NULL) {
                    fprintf(stderr, "sync error before RDMA ops\n");
                    return -1;
                }
                if (sock_write_data(res->sock)) { /* just send a dummy char back and forth */
                    fprintf(stderr, "sync error before RDMA ops\n");
                    return -1;
                }

                // fetching MatrixPower
                sock_write_request(res->req_sock, id, j, i, SUB_M, 1, 1);
                sock_read_data(res->req_sock);
                entry->power_resp = sock_read_offset(res->sock);
                if (entry->power_resp == NULL) {
                    fprintf(stderr, "sync error before RDMA ops\n");
                    return -1;
                }
    
                if (sock_write_data(res->sock)) { /* just send a dummy char back and forth */
                    fprintf(stderr, "sync error before RDMA ops\n");
                    return -1;
                }
                timing_info_push_end(fetch_timing);

                timing_info_push_start(copy_in_timing);
                cudaMemcpy(MatrixTempIn, res->buf + entry->temp_resp->offset, dsize * sizeof(double), cudaMemcpyHostToDevice);
                free(entry->temp_resp);
                cudaMemcpy(MatrixPower, res->buf + entry->power_resp->offset, dsize * sizeof(double), cudaMemcpyHostToDevice);
                free(entry->power_resp);
                fifo_push(complete_queue, entry);
                timing_info_push_end(copy_in_timing);

                timing_info_push_start(kernel_timing);
                calculate_temp<<<dimGrid, dimBlock>>>(MIN(num_iterations, total_iterations-t), MatrixPower, MatrixTempIn, MatrixTempOut,
                    grid_rows, grid_cols, borderCols, borderRows, Cap, Rx, Ry, Rz, step, time_elapsed);
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("Kernel Error: %s\n", cudaGetErrorString(err));
                }
            
                timing_info_push_end(kernel_timing);
        
                timing_info_push_start(copy_out_timing);
                cudaMemcpy(MatrixOut, MatrixTempOut, dsize * sizeof(double), cudaMemcpyDeviceToHost);
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("cudaMemcpy Error: %s\n", cudaGetErrorString(err));
                }
                timing_info_push_end(copy_out_timing);
            }
        }
    }
    sock_write_request(res->req_sock, -1, j, i, SUB_M, 1, 0);
    sock_read_data(res->req_sock);
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    printf("Ending simulation\n");
    printf("GEMM duration: %f ms\n", (float) duration / 1000);    

    printf("Fetch time: %f ms\n", (float) timing_info_duration(fetch_timing) / 1000);
    printf("Copy in time: %f ms\n", (float) timing_info_duration(copy_in_timing) / 1000);
    printf("memset time: %f ms\n", (float) timing_info_duration(memset_timing) / 1000);
    printf("convolution row time: %f ms\n", (float) timing_info_duration(kernel_timing) / 1000);
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

    tss = timing_info_get_timestamps(copy_in_timing);
    fptr = fopen("copy_in_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(copy_in_timing);

    tss = timing_info_get_timestamps(memset_timing);
    fptr = fopen("memset_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(memset_timing);

    tss = timing_info_get_timestamps(kernel_timing);
    fptr = fopen("convolution_row_ts.bin", "wb");
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

    // for (i = 0; i < 16; i++) {
    //     printf("%f ", MatrixOut[i]);
    // }
    // printf("\n");
    cudaFree(MatrixPower);
    cudaFree(MatrixTempIn);
    cudaFree(MatrixTempOut);
    free(MatrixOut);

    fifo_free(sending_queue);
    fifo_free(complete_queue);
    free(entries);    
    cudaDeviceReset();

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

    fprintf(stdout, "running server\n");

    printf("calculating the result of SPDK GEMM\n");
    rc = spdk_nds_hotspot(&res, matrix_id, n, sub_n);

    // config_destroy(&config);

    munmap(hugepage_addr, BUF_SIZE);
    
    close(res.sock);
    close(res.req_sock);
    munmap(hugepage_addr, BUF_SIZE);
    close(hugepage_fd);

    printf("test result is %i\n", rc);
    return rc;
}
