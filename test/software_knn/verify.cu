#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <cuda.h>
#include "cublas_v2.h"

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

extern "C" {
    #include "rdma.h"
    #include "timing.h"
    #include "fifo.h"
}

#define BLOCK_DIM 16UL
#define THREADS_PER_BLOCK (BLOCK_DIM*BLOCK_DIM)

#define HUGEPAGE_SZ (4UL * 1024UL * 1024UL * 1024UL)
#define M 65536UL
#define SUB_M 4096UL
#define AGGREGATED_SZ (M * SUB_M * 8UL)

// #define IO_QUEUE_SZ (HUGEPAGE_SZ / AGGREGATED_SZ)
#define IO_QUEUE_SZ 1UL

void print_config(struct config_t config);

struct fetch_conf {
    struct resources *res;
    uint64_t m, sub_m;
    double *ref_dev;
    char *hugepage_addr;
    struct fifo *sending_queue;
    struct fifo *complete_queue;
    struct timing_info *col_fetch_timing;
    struct timing_info *copy_in_timing;
};

struct request_conf {
    struct resources *res;
    uint64_t id;
    uint64_t sub_m;
};

struct fifo_entry {
    double *ref_dev;
};

/**
 * Computes the squared norm of each column of the input array.
 *
 * @param array   input array
 * @param width   number of columns of `array` = number of points
 * @param pitch   pitch of `array` in number of columns
 * @param height  number of rows of `array` = dimension of the points
 * @param norm    output array containing the squared norm values
 */
 __global__ void compute_squared_norm(double * array, uint64_t width, uint64_t pitch, uint64_t height, double * norm){
    uint64_t xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (xIndex<width){
        double sum = 0.f;
        for (uint64_t i=0; i<height; i++){
            double val = array[i*pitch+xIndex];
            sum += val*val;
        }
        norm[xIndex] = sum;
    }
}

/**
 * Add the reference points norm (column vector) to each colum of the input array.
 *
 * @param array   input array
 * @param width   number of columns of `array` = number of points
 * @param pitch   pitch of `array` in number of columns
 * @param height  number of rows of `array` = dimension of the points
 * @param norm    reference points norm stored as a column vector
 */
 __global__ void add_reference_points_norm(double * array, uint64_t width, uint64_t pitch, uint64_t height, double * norm){
    uint64_t tx = threadIdx.x;
    uint64_t ty = threadIdx.y;
    uint64_t xIndex = blockIdx.x * blockDim.x + tx;
    uint64_t yIndex = blockIdx.y * blockDim.y + ty;
    __shared__ double shared_vec[BLOCK_DIM];
    if (tx==0 && yIndex<height)
        shared_vec[ty] = norm[yIndex];
    __syncthreads();
    if (xIndex<width && yIndex<height)
        array[yIndex*pitch+xIndex] += shared_vec[ty];
}

/**
 * For each reference point (i.e. each column) finds the k-th smallest distances
 * of the distance matrix and their respective indexes and gathers them at the top
 * of the 2 arrays.
 *
 * Since we only need to locate the k smallest distances, sorting the entire array
 * would not be very efficient if k is relatively small. Instead, we perform a
 * simple insertion sort by eventually inserting a given distance in the first
 * k values.
 *
 * @param dist         distance matrix
 * @param dist_pitch   pitch of the distance matrix given in number of columns
 * @param index        index matrix
 * @param index_pitch  pitch of the index matrix given in number of columns
 * @param width        width of the distance matrix and of the index matrix
 * @param height       height of the distance matrix
 * @param k            number of values to find
 */
 __global__ void modified_insertion_sort(double * dist,
    uint64_t     dist_pitch,
    int64_t *   index,
    uint64_t     index_pitch,
    uint64_t     width,
    uint64_t     height,
    uint64_t     k)
{
    // Column position
    uint64_t xIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Do nothing if we are out of bounds
    if (xIndex < width) {

        // Pointer shift
        double * p_dist  = dist  + xIndex;
        int64_t *   p_index = index + xIndex;

        // Initialise the first index
        p_index[0] = 0;

        // Go through all points
        for (uint64_t i=1; i<height; ++i) {

            // Store current distance and associated index
            double curr_dist = p_dist[i*dist_pitch];
            uint64_t   curr_index  = i;

            // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
            if (i >= k && curr_dist >= p_dist[(k-1)*dist_pitch]) {
                continue;
            }

            // Shift values (and indexes) higher that the current distance to the right
            uint64_t j = min(i, k-1);
            while (j > 0 && p_dist[(j-1)*dist_pitch] > curr_dist) {
                p_dist[j*dist_pitch]   = p_dist[(j-1)*dist_pitch];
                p_index[j*index_pitch] = p_index[(j-1)*index_pitch];
                --j;
            }

            // Write the current distance and index at their position
            p_dist[j*dist_pitch]   = curr_dist;
            p_index[j*index_pitch] = curr_index; 
        }
    }
}

/**
 * Adds the query points norm (row vector) to the k first lines of the input
 * array and computes the square root of the resulting values.
 *
 * @param array   input array
 * @param width   number of columns of `array` = number of points
 * @param pitch   pitch of `array` in number of columns
 * @param k       number of neighbors to consider
 * @param norm     query points norm stored as a row vector
 */
 __global__ void add_query_points_norm_and_sqrt(double * array, uint64_t width, uint64_t pitch, uint64_t k, double * norm){
    uint64_t xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex<width && yIndex<k)
        array[yIndex*pitch + xIndex] = sqrt(array[yIndex*pitch + xIndex] + norm[xIndex]);
}


int cudaMemcpyFromMmap(struct fetch_conf *conf, char *dst, const char *src, const size_t length, struct timing_info *fetch_timing) {
    struct response *res = NULL;

    timing_info_push_start(fetch_timing);
    res = sock_read_offset(conf->res->sock);
    if (res == NULL) {
        fprintf(stderr, "sync error before RDMA ops\n");
        return 1;
    }

    // if (res->id == 0) {
    //     printf("fetching row [%lu:%lu]\n", res->x, res->y);
    // } else {
    //     printf("fetching col [%lu:%lu]\n", res->x, res->y);
    // }
    // printf("offset: %lu\n", res->offset);

    timing_info_push_end(fetch_timing);

    timing_info_push_start(conf->copy_in_timing);
    cudaMemcpy(dst, src + res->offset, length, cudaMemcpyHostToDevice);
    timing_info_push_end(conf->copy_in_timing);

    free(res);
    if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
        fprintf(stderr, "sync error before RDMA ops\n");
        return 1;
    }
    return 0;
}

void *fetch_thread(void *args) {
    struct fetch_conf *conf = (struct fetch_conf *) args;
    uint64_t st;
    uint64_t dsize = conf->m * conf->sub_m;
    double *ptr_a;
    struct fifo_entry *entry = NULL;
    uint64_t count = 0;

    for (st = 0; st < conf->m / conf->sub_m; st++) {
        entry = (struct fifo_entry *) fifo_pop(conf->complete_queue);
        ptr_a = conf->ref_dev + dsize * (count % IO_QUEUE_SZ);

        cudaMemcpyFromMmap(conf, (char *) ptr_a, (char *) conf->hugepage_addr, dsize * sizeof(double), conf->col_fetch_timing);
        count++;

        entry->ref_dev = ptr_a;
        fifo_push(conf->sending_queue, entry);
    }
    return NULL;
}

void *request_thread(void *args) {
    struct request_conf *conf = (struct request_conf *) args;
    uint64_t st;

    for (st = 0; st < M / SUB_M; st++) {
        sock_write_request(conf->res->req_sock, conf->id, st, st+1, conf->sub_m, 3, 0);
        sock_read_data(conf->res->req_sock);
    }

    // send a signal to tell storage backend the iteration is done.
    sock_write_request(conf->res->req_sock, -1, st, st+1, SUB_M, 0, 0);
    sock_read_data(conf->res->req_sock);
    return NULL;
}

int nds_knn(struct resources *res, uint64_t id, uint64_t ref_nb, uint64_t subref_nb, const double *query, uint64_t query_nb, uint64_t k) {
    // Return variables
    cudaError_t err0, err1, err2, err3, err4, err5;

    // Check that we have at least one CUDA device 
    int nb_devices;
    err0 = cudaGetDeviceCount(&nb_devices);
    if (err0 != cudaSuccess || nb_devices == 0) {
        printf("ERROR: No CUDA device found\n");
        return -1;
    }

    // Select the first CUDA device as default
    err0 = cudaSetDevice(0);
    if (err0 != cudaSuccess) {
        printf("ERROR: Cannot set the chosen CUDA device\n");
        return -1;
    }

    // Allocate input points and output k-NN distances / indexes
    double *knn_dist = (double *) malloc(query_nb * k * sizeof(double));
    int64_t *knn_index = (int64_t *) malloc(query_nb * k * sizeof(int64_t));

    // Allocation checks
    if (!knn_dist || !knn_index) {
        printf("Error: Memory allocation error\n"); 
        free(knn_dist);
        free(knn_index);
        return EXIT_FAILURE;
    }

    // Initialize CUBLAS
    cublasStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    // Allocate global memory
    double *ref_dev, *query_dev, *dist_dev, *ref_norm_dev, *query_norm_dev;
    int64_t *index_dev = NULL;
    size_t dim = ref_nb;
    err0 = cudaMalloc((void**) &ref_dev, SUB_M * sizeof(double) * dim * IO_QUEUE_SZ); // 4 GB
    err1 = cudaMalloc((void**) &query_dev, query_nb * sizeof(double) * dim); // 2 GB
    err2 = cudaMalloc((void**) &dist_dev, query_nb * sizeof(double) * ref_nb); // 
    err3 = cudaMalloc((void**) &index_dev, query_nb * sizeof(int64_t) * k);
    err4 = cudaMalloc((void**) &ref_norm_dev, ref_nb * sizeof(double));
    err5 = cudaMalloc((void**) &query_norm_dev, query_nb * sizeof(double));
    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess || err5 != cudaSuccess) {
    printf("ERROR: Memory allocation error\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasDestroy(handle);
        return -1;
    }

    // Deduce pitch values
    size_t ref_pitch   = SUB_M;
    size_t query_pitch = query_nb;
    size_t dist_pitch  = query_nb;
    size_t index_pitch = query_nb;

    // Check pitch values
    if (query_pitch != dist_pitch || query_pitch != index_pitch) {
        printf("ERROR: Invalid pitch value\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasDestroy(handle);
        return -1; 
    }

    size_t i;

    struct fifo *sending_queue;
    struct fifo *complete_queue; 
    struct fifo_entry *entries = (struct fifo_entry *) calloc(IO_QUEUE_SZ, sizeof(struct fifo_entry));
    struct fifo_entry *entry = NULL;

    struct timing_info *queue_timing;
    struct timing_info *col_fetch_timing;
    struct timing_info *copy_in_timing;
    struct timing_info *kernel_timing;
    struct timing_info *copy_out_timing;    
    
    pthread_t f_thread_id; 
    struct fetch_conf f_conf;

    pthread_t r_thread_id; 
    struct request_conf r_conf;

    struct timeval h_start, h_end;
    long duration;

    // initialization
    size_t total_iteration = (ref_nb / subref_nb);
    queue_timing = timing_info_new(total_iteration);
    if (queue_timing == NULL) {
        printf("cannot create queue_timing\n");
        return -1;
    }

    col_fetch_timing = timing_info_new(total_iteration);
    if (col_fetch_timing == NULL) {
        printf("cannot create col_fetch_timing\n");
        return -1;
    }

    copy_in_timing = timing_info_new(total_iteration * 2);
    if (copy_in_timing == NULL) {
        printf("cannot create copy_in_timing\n");
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

    r_conf.res = res;
    r_conf.id = id;
    r_conf.sub_m = SUB_M;

    // create thread here
    f_conf.res = res;
    f_conf.m = ref_nb;
    f_conf.sub_m = subref_nb;
    f_conf.ref_dev = ref_dev;
    f_conf.hugepage_addr = res->buf;
    f_conf.sending_queue = sending_queue;
    f_conf.complete_queue = complete_queue;
    f_conf.col_fetch_timing = col_fetch_timing;
    f_conf.copy_in_timing = copy_in_timing;

    timing_info_set_starting_time(queue_timing);
    timing_info_set_starting_time(col_fetch_timing);
    timing_info_set_starting_time(copy_in_timing);
    timing_info_set_starting_time(kernel_timing);
    timing_info_set_starting_time(copy_out_timing);

    gettimeofday(&h_start, NULL);
    pthread_create(&r_thread_id, NULL, request_thread, &r_conf); 
    // pthread_create(&f_thread_id, NULL, fetch_thread, &f_conf); 

    // Copy reference and query data from the host to the device
    err1 = cudaMemcpy(query_dev, query, query_nb * sizeof(double) * dim, cudaMemcpyHostToDevice);
    if (err1 != cudaSuccess) {
        printf("ERROR: Unable to copy data from host to device\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasDestroy(handle);
        return -1; 
    }

    // Compute the squared norm of the query points
    compute_squared_norm<<<(query_nb+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(query_dev, query_nb, query_pitch, dim, query_norm_dev);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasDestroy(handle);
        return -1;
    }

    double alpha = -2.0, beta = 0.0;
    double *ptr_a;
    uint64_t dsize = M * SUB_M;
    uint64_t count = 0;
    // blockGEMM
    for (i = 0; i < ref_nb; i+=SUB_M) {
        // printf("i: %lu\n", i);
        timing_info_push_start(queue_timing);
        ptr_a = ref_dev + dsize * (count % IO_QUEUE_SZ);

        cudaMemcpyFromMmap(&f_conf, (char *) ptr_a, (char *) res->buf, dsize * sizeof(double), col_fetch_timing);
        count++;

        // ptr_a = ptr_a;
        // entry = (struct fifo_entry *) fifo_pop(sending_queue);
        timing_info_push_end(queue_timing);

        timing_info_push_start(kernel_timing);
        // Compute the squared norm of the reference points
        compute_squared_norm<<<(SUB_M+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(ptr_a, SUB_M, ref_pitch, dim, ref_norm_dev);
        if (cudaGetLastError() != cudaSuccess) {
            printf("ERROR: Unable to execute kernel\n");
            cudaFree(ref_dev);
            cudaFree(query_dev);
            cudaFree(dist_dev);
            cudaFree(index_dev);
            cudaFree(ref_norm_dev);
            cudaFree(query_norm_dev);
            cublasDestroy(handle);
            return -1;
        }
        // Computation of query*transpose(reference)
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, query_pitch, ref_pitch, dim, &alpha, query_dev, query_pitch, ptr_a, ref_pitch, &beta, dist_dev + i*query_nb, query_pitch);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("ERROR: Unable to execute cublasSgemm\n");
            cudaFree(ref_dev);
            cudaFree(query_dev);
            cudaFree(dist_dev);
            cudaFree(index_dev);
            cudaFree(ref_norm_dev);
            cudaFree(query_norm_dev);
            cublasDestroy(handle);
            return -1;       
        }

        // Add reference points norm
        dim3 block2(BLOCK_DIM, BLOCK_DIM, 1);
        dim3 grid2((query_nb+BLOCK_DIM-1) / BLOCK_DIM, (SUB_M+BLOCK_DIM-1) / BLOCK_DIM, 1);
        add_reference_points_norm<<<grid2, block2>>>(dist_dev + i*query_nb, query_nb, dist_pitch, SUB_M, ref_norm_dev);
        if (cudaGetLastError() != cudaSuccess) {
            printf("ERROR: Unable to execute kernel\n");
            cudaFree(ref_dev);
            cudaFree(query_dev);
            cudaFree(dist_dev);
            cudaFree(index_dev);
            cudaFree(ref_norm_dev);
            cudaFree(query_norm_dev);
            cublasDestroy(handle);
            return -1;
        }
        // fifo_push(complete_queue, entry);
        cudaDeviceSynchronize();
        timing_info_push_end(kernel_timing);
    }

    // Sort each column for the top-k results
    modified_insertion_sort<<<(query_nb+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(dist_dev, dist_pitch, index_dev, index_pitch, query_nb, ref_nb, k);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasDestroy(handle);
        return -1;
    }

    // Add query norm and compute the square root of the of the k first elements
    dim3 block3(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid3((query_nb+BLOCK_DIM-1) / BLOCK_DIM, (k+BLOCK_DIM-1) / BLOCK_DIM, 1);
    add_query_points_norm_and_sqrt<<<grid3, block3>>>(dist_dev, query_nb, dist_pitch, k, query_norm_dev);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasDestroy(handle);
        return -1;
    }

    timing_info_push_start(copy_out_timing);
    // Copy k smallest distances / indexes from the device to the host
    err0 = cudaMemcpy(knn_dist, dist_dev, query_nb * sizeof(double) * k, cudaMemcpyDeviceToHost);
    err1 = cudaMemcpy(knn_index, index_dev, query_nb * sizeof(int64_t) * k, cudaMemcpyDeviceToHost);
    if (err0 != cudaSuccess || err1 != cudaSuccess) {
        printf("ERROR: Unable to copy data from device to host\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasDestroy(handle);
        return -1; 
    }
    timing_info_push_end(copy_out_timing);

    pthread_join(r_thread_id, NULL); 
    // pthread_join(f_thread_id, NULL); 

    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    
    printf("KNN End-to-end duration: %f ms\n", (float) duration / 1000);    
    printf("Col fetch time: %f ms\n", (float) timing_info_duration(col_fetch_timing) / 1000);
    printf("Copy in time: %f ms\n", (float) timing_info_duration(copy_in_timing) / 1000);
    printf("sending_queue waiting time: %f ms\n", (float) timing_info_duration(queue_timing) / 1000);
    printf("Kernel time: %f ms\n", (float) timing_info_duration(kernel_timing) / 1000);
    printf("copy out time: %f ms\n", (float) timing_info_duration(copy_out_timing) / 1000);
    
    struct timestamps *tss = NULL;
    FILE *fptr;
    tss = timing_info_get_timestamps(col_fetch_timing);
    fptr = fopen("col_fetch_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(col_fetch_timing);

    tss = timing_info_get_timestamps(copy_in_timing);
    fptr = fopen("copy_in_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(copy_in_timing);
    
    tss = timing_info_get_timestamps(queue_timing);
    fptr = fopen("queue_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(queue_timing);

    tss = timing_info_get_timestamps(kernel_timing);
    fptr = fopen("kernel_ts.bin", "wb");
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

    fptr = fopen("knn_dist.bin", "wb");
    fwrite(knn_dist, sizeof(double), query_nb * k, fptr);
    fclose(fptr);
    
    fptr = fopen("knn_index.bin", "wb");
    fwrite(knn_index, sizeof(int64_t), query_nb * k, fptr);
    fclose(fptr);

    // Memory clean-up and CUBLAS shutdown
    free(knn_dist);
    free(knn_index);
    cudaFree(ref_dev);
    cudaFree(query_dev);
    cudaFree(dist_dev);
    cudaFree(index_dev);
    cudaFree(ref_norm_dev);
    cudaFree(query_norm_dev);
    cublasDestroy(handle);
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
    uint64_t matrix_id, n, sub_n, query_nb, k, dim;
    double *query;
    int hugepage_fd, query_fd;
    char *hugepage_addr;
    double *query_addr;

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
    if (argc < 8) {
        printf("usage: %s <query_path> <matrix_id> <# of ref points> <# of sub-ref points> <# of query points> <# of k> <port>\n", argv[0]);
        exit(1);
    } 
    matrix_id = (uint64_t) atoll(argv[2]);
    n = (uint64_t) atoll(argv[3]);
    sub_n = (uint64_t) atoll(argv[4]);
    query_nb = (uint64_t) atoll(argv[5]);
    k = (uint64_t) atoll(argv[6]);
    config.tcp_port = atoi(argv[7]);

    // assume # of attributes is the same as # of ref points
    dim = n;

    /* print the used parameters for info*/
    print_config(config);
    
    // initialize data for query points here.
    query_fd = open(argv[1], O_RDONLY);
    if (query_fd < 0) {
        perror("open");
        exit(1);
    }

    query_addr = (double *) mmap(0, dim * query_nb * sizeof(double), PROT_READ, MAP_PRIVATE, query_fd, 0);
    if (query_addr==MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    // data layout is [ref_nb][point] based on the description in repo.
    query = (double *) malloc(dim * query_nb * sizeof(double));    
    if (query == NULL) {
        printf("Error: Memory allocation error\n"); 
        return EXIT_FAILURE;
    }

    memcpy(query, query_addr, dim * query_nb * sizeof(double));
    munmap(query_addr, dim * query_nb * sizeof(double));
    close(query_fd);

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

    printf("calculating the result of pagerank\n");
    rc = nds_knn(&res, matrix_id, n, sub_n, query, query_nb, k);
    
    close(res.sock);
    close(res.req_sock);
    munmap(hugepage_addr, BUF_SIZE);
    close(hugepage_fd);
    free(query);
    return rc;
}
