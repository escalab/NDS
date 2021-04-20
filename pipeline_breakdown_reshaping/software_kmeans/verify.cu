#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <cuda.h>
#include <assert.h>

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
#define SUB_M 1024UL
#define AGGREGATED_SZ (M * SUB_M * 8UL)

// #define IO_QUEUE_SZ (HUGEPAGE_SZ / AGGREGATED_SZ)
#define IO_QUEUE_SZ 1UL

#define msg(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); } while (0)
#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        // cudaGetErrorString() isn't always very helpful. Look up the error
        // number in the cudaError enum in driver_types.h in the CUDA includes
        // directory for a better explanation.
        err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif

#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)

void print_config(struct config_t config);

struct fetch_conf {
    struct resources *res;
    uint64_t m, sub_m;
    double *deviceObjects;
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
    double *deviceObjects;
};

void col_reshape(double *dst, double *src, size_t m, size_t sub_m) {
    uint64_t i;
    double *dst_ptr = dst, *src_ptr = src;

    for (i = 0; i < sub_m; i++) {
        memcpy(dst_ptr, src_ptr, sizeof(double) * sub_m);
        dst_ptr += sub_m;
        src_ptr += m;
    }
}

void colstripe_reassemble(double *dst, double *src, size_t m, size_t sub_m, size_t granularity) {
    uint64_t chunk, i, row, col;
    uint64_t dsize = m * sub_m;
    uint64_t multiplier = sub_m / granularity;
    double *dst_ptr, *src_ptr = src;
    for (chunk = 0; chunk < (dsize / granularity / granularity); chunk++) {
        row = chunk / multiplier;
        col = chunk % multiplier;
        dst_ptr = dst + row * sub_m * granularity + col * granularity; 
        for (i = 0; i < granularity; i++) {
            memcpy(dst_ptr, src_ptr, sizeof(double) * granularity);
            dst_ptr += sub_m;
            src_ptr += granularity;
        }
    }
}

static inline uint64_t nextPowerOfTwo(uint64_t n) {
    n--;

    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;
//  n = n >> 32 | n;    //  For 64-bit ints

    return ++n;
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__host__ __device__ inline static
double euclid_dist_2(uint64_t    numCoords,
                    uint64_t    numObjs,
                    uint64_t    numClusters,
                    double *objects,     // [numCoords][numObjs]
                    double *clusters,    // [numCoords][numClusters]
                    int    objectId,
                    int    clusterId)
{
    uint64_t i;
    double ans=0.0;

    for (i = 0; i < numCoords; i++) {
        ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
               (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
    }

    return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__global__ static
void find_nearest_cluster_block(int numCoords,
                          int numObjs,
                          int numClusters,
                          int offset,
                          double *objects,           //  [numCoords][numObjs]
                          double *deviceClusters,    //  [numCoords][numClusters]
                          int *membership,          //  [numObjs]
                          int *intermediates)
{
    extern __shared__ char sharedMemory[];

    //  The type chosen for membershipChanged must be large enough to support
    //  reductions! There are blockDim.x elements, one for each thread in the
    //  block. See numThreadsPerClusterBlock in cuda_kmeans().
    unsigned char *membershipChanged = (unsigned char *)sharedMemory;
#if BLOCK_SHARED_MEM_OPTIMIZATION
    double *clusters = (double *)(sharedMemory + blockDim.x);
#else
    double *clusters = deviceClusters;
#endif

    membershipChanged[threadIdx.x] = 0;

#if BLOCK_SHARED_MEM_OPTIMIZATION
    //  BEWARE: We can overrun our shared memory here if there are too many
    //  clusters or too many coordinates! For reference, a Tesla C1060 has 16
    //  KiB of shared memory per block, and a GeForce GTX 480 has 48 KiB of
    //  shared memory per block.
    for (int i = threadIdx.x; i < numClusters; i += blockDim.x) {
        for (int j = 0; j < numCoords; j++) {
            clusters[numClusters * j + i] = deviceClusters[numClusters * j + i];
        }
    }
    __syncthreads();
#endif

    int objectId = blockDim.x * blockIdx.x + threadIdx.x;

    if (objectId < numObjs) {
        int   index, i;
        double dist, min_dist;

        /* find the cluster id that has min distance to object */
        index    = 0;
        min_dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, 0);

        for (i=1; i<numClusters; i++) {
            dist = euclid_dist_2(numCoords, numObjs, numClusters,
                                 objects, clusters, objectId, i);
            /* no need square root */
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = i;
            }
        }

        if (membership[objectId + offset] != index) {
            membershipChanged[threadIdx.x] = 1;
        }

        /* assign the membership to object objectId */
        membership[objectId + offset] = index;

        __syncthreads();    //  For membershipChanged[]

        //  blockDim.x *must* be a power of two!
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                membershipChanged[threadIdx.x] +=
                    membershipChanged[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            intermediates[blockIdx.x] += membershipChanged[0];
        }
    }
}

__global__ static
void compute_delta(int *deviceIntermediates,
                   int numIntermediates,    //  The actual number of intermediates
                   int numIntermediates2)   //  The next power of two
{
    //  The number of elements in this array should be equal to
    //  numIntermediates2, the number of threads launched. It *must* be a power
    //  of two!
    extern __shared__ unsigned int intermediates[];

    //  Copy global intermediate values into shared memory.
    intermediates[threadIdx.x] =
        (threadIdx.x < numIntermediates) ? deviceIntermediates[threadIdx.x] : 0;

    __syncthreads();

    //  numIntermediates2 *must* be a power of two!
    for (unsigned int s = numIntermediates2 / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            intermediates[threadIdx.x] += intermediates[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        deviceIntermediates[0] = intermediates[0];
    }
}

/*---< file_write() >---------------------------------------------------------*/
int file_write(uint64_t        numClusters,  /* no. clusters */
    uint64_t        numObjs,      /* no. data objects */
    uint64_t        numCoords,    /* no. coordinates (local) */
    double    **clusters,     /* [numClusters][numCoords] centers */
    int       *membership)   /* [numObjs] */
{
    FILE *fptr;
    int   i, j;

    /* output: the coordinates of the cluster centres ----------------------*/
    printf("Writing coordinates of K=%lu cluster centers to file \"%s\"\n", numClusters, "cluster_centres");
    fptr = fopen("cluster_centres", "w");
    for (i=0; i<numClusters; i++) {
        fprintf(fptr, "%d ", i);
        for (j=0; j<numCoords; j++) {
            fprintf(fptr, "%f ", clusters[i][j]);
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);

    /* output: the closest cluster centre to each of the data points --------*/
    printf("Writing membership of N=%lu data objects to file \"%s\"\n", numObjs, "membership");
    fptr = fopen("membership", "w");
    for (i=0; i<numObjs; i++) {
        fprintf(fptr, "%d %d\n", i, membership[i]);
    }
    fclose(fptr);

    return 0;
}

int cudaMemcpyFromMmap(struct fetch_conf *conf, uint64_t id, uint64_t st, uint64_t en, uint64_t size, uint64_t op, uint64_t which,
    char *dst, const char *src, const size_t length, struct timing_info *fetch_timing) {
    struct response *res = NULL;

    timing_info_push_start(fetch_timing);
    sock_write_request(conf->res->req_sock, id, st, en, size, op, which);
    sock_read_data(conf->res->req_sock);

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

double** nds_kmeans_block(struct resources *res, uint64_t id, uint64_t numCoords, uint64_t numObjs, 
    uint64_t numClusters, uint64_t sub_numObjs, double threshold, int *membership, int *loop_iterations) {
    int      i, j, index, loop=0;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                    new cluster */
    double    delta;          /* % of objects change their clusters */
    double  **clusters;       /* out: [numClusters][numCoords] */
    double  **dimClusters;
    double  **newClusters;    /* [numCoords][numClusters] */

    double *deviceObjects;
    double *deviceClusters;
    int *deviceMembership;
    int *deviceIntermediates;
    
    struct response *resp = NULL;
    double *reshaped_data = (double *) calloc(M * SUB_M, sizeof(double));
    uint64_t sub_st;
    double *src_ptr;
    double *buf;
    
    /* pick first numClusters elements of objects[] as initial cluster centers*/
    malloc2D(clusters, numClusters, numCoords, double);
    malloc2D(dimClusters, numCoords, numClusters, double);

    sock_write_request(res->req_sock, id, 0, 1, SUB_M, 3, 0);
    sock_read_data(res->req_sock);

    resp = sock_read_offset(res->sock);
    if (resp == NULL) {
        fprintf(stderr, "sync error before RDMA ops\n");
        return clusters;
    }

    // testing SEQ format so we don't need to reshape here
    // colstripe_reassemble(reshaped_data, (double *) (res->buf + resp->offset), M, SUB_M, 256UL);

    buf = (double *) (res->buf + resp->offset);
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = buf[i * SUB_M + j];
        }
    }

    free(resp);
    if (sock_write_data(res->sock)) { /* just send a dummy char back and forth */
        fprintf(stderr, "sync error before RDMA ops\n");
        return clusters;
    }

    /* initialize membership[] */
    for (i=0; i<numObjs; i++) {
        membership[i] = -1;
    }
    
    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    malloc2D(newClusters, numCoords, numClusters, double);
    memset(newClusters[0], 0, numCoords * numClusters * sizeof(double));

    //  To support reduction, numThreadsPerClusterBlock *must* be a power of
    //  two, and it *must* be no larger than the number of bits that will
    //  fit into an unsigned char, the type used to keep track of membership
    //  changes in the kernel.
    const unsigned int numThreadsPerClusterBlock = 128;
    const unsigned int numClusterBlocks =
        (SUB_M + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;
#if BLOCK_SHARED_MEM_OPTIMIZATION
    const unsigned int clusterBlockSharedDataSize =
        numThreadsPerClusterBlock * sizeof(unsigned char) +
        numClusters * numCoords * sizeof(double);

    cudaDeviceProp deviceProp;
    int deviceNum;
    cudaGetDevice(&deviceNum);
    cudaGetDeviceProperties(&deviceProp, deviceNum);

    if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
        err("WARNING: Your CUDA hardware has insufficient block shared memory. "
        "You need to recompile with BLOCK_SHARED_MEM_OPTIMIZATION=0. "
        "See the README for details.\n");
    }
#else
    const unsigned int clusterBlockSharedDataSize =
    numThreadsPerClusterBlock * sizeof(unsigned char);
#endif

    const unsigned int numReductionThreads =
        nextPowerOfTwo(numClusterBlocks);
    const unsigned int reductionBlockSharedDataSize =
        numReductionThreads * sizeof(unsigned int);

    checkCuda(cudaMalloc(&deviceObjects, SUB_M*numCoords*sizeof(double)*IO_QUEUE_SZ));
    checkCuda(cudaMalloc(&deviceClusters, numClusters*numCoords*sizeof(double)));
    checkCuda(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
    checkCuda(cudaMalloc(&deviceIntermediates, numReductionThreads*sizeof(unsigned int)));

    checkCuda(cudaMemcpy(deviceMembership, membership, numObjs*sizeof(int), cudaMemcpyHostToDevice));

    struct timing_info *col_fetch_timing;
    struct timing_info *reshape_timing;    
    struct timing_info *copy_in_timing;
    struct timing_info *kernel_timing;
    struct timing_info *copy_out_timing;
    struct timing_info *cluster_fetch_timing;   
    struct timing_info *cluster_reshape_timing;
    struct timing_info *cluster_assignment_timing;    

    struct fetch_conf f_conf;

    struct timeval h_start, h_end;
    long duration;

    // initialization
    // tested beforehand and found out we will have 11 loops;
    size_t total_iteration = (M / SUB_M) * 32;

    col_fetch_timing = timing_info_new(total_iteration * (M / SUB_M));
    if (col_fetch_timing == NULL) {
        printf("cannot create col_fetch_timing\n");
        return clusters;
    }

    reshape_timing = timing_info_new(total_iteration * (M / SUB_M));
    if (reshape_timing == NULL) {
        printf("cannot create reshape_timing\n");
        return clusters;
    }

    copy_in_timing = timing_info_new(total_iteration);
    if (copy_in_timing == NULL) {
        printf("cannot create copy_in_timing\n");
        return clusters;
    }

    kernel_timing = timing_info_new(total_iteration);
    if (kernel_timing == NULL) {
        printf("cannot create kernel_timing\n");
        return clusters;
    }

    copy_out_timing = timing_info_new(total_iteration);
    if (copy_out_timing == NULL) {
        printf("cannot create copy_out_timing\n");
        return clusters;
    }

    cluster_fetch_timing = timing_info_new(total_iteration * (M / SUB_M));
    if (cluster_fetch_timing == NULL) {
        printf("cannot create cluster_fetch_timing\n");
        return clusters;
    }

    cluster_reshape_timing = timing_info_new(total_iteration * (M / SUB_M));
    if (reshape_timing == NULL) {
        printf("cannot create cluster_reshape_timing\n");
        return clusters;
    }

    cluster_assignment_timing = timing_info_new(total_iteration);
    if (cluster_assignment_timing == NULL) {
        printf("cannot create cluster_assignment_timing\n");
        return clusters;
    }

    // create thread here
    f_conf.res = res;
    f_conf.m = M;
    f_conf.sub_m = SUB_M;
    f_conf.deviceObjects = deviceObjects;
    f_conf.hugepage_addr = res->buf;
    f_conf.col_fetch_timing = col_fetch_timing;
    f_conf.copy_in_timing = copy_in_timing;
    
    timing_info_set_starting_time(col_fetch_timing);
    timing_info_set_starting_time(reshape_timing);
    timing_info_set_starting_time(copy_in_timing);
    timing_info_set_starting_time(kernel_timing);
    timing_info_set_starting_time(copy_out_timing);
    timing_info_set_starting_time(cluster_fetch_timing);
    timing_info_set_starting_time(cluster_reshape_timing);
    timing_info_set_starting_time(cluster_assignment_timing);

    // computation part
    uint64_t st;
    gettimeofday(&h_start, NULL);
    do {
        checkCuda(cudaMemset(deviceIntermediates, 0, numReductionThreads*sizeof(unsigned int)));
        checkCuda(cudaMemcpy(deviceClusters, dimClusters[0], numClusters*numCoords*sizeof(double), cudaMemcpyHostToDevice));
        
        for (i = 0, st = 0; i < numObjs; i += SUB_M, st++) {
            printf("i: %d\n", i);
            // TODO: need cudaMemcpy2D
            for (sub_st = 0; sub_st < M / SUB_M; sub_st++) {
                timing_info_push_start(col_fetch_timing);
                sock_write_request(res->req_sock, id, sub_st, sub_st+1, SUB_M, 2, 0);
                sock_read_data(res->req_sock);
            
                resp = sock_read_offset(res->sock);
                if (resp == NULL) {
                    fprintf(stderr, "sync error before RDMA ops\n");
                    return clusters;
                }
                timing_info_push_end(col_fetch_timing);
                
                timing_info_push_start(reshape_timing);
                src_ptr = (double *) (res->buf + resp->offset);
                col_reshape(reshaped_data + sub_st * SUB_M * SUB_M, src_ptr + i, M, SUB_M);
                if (sock_write_data(res->sock)) { /* just send a dummy char back and forth */
                    fprintf(stderr, "sync error before RDMA ops\n");
                    return clusters;
                }
                free(resp);
                timing_info_push_end(reshape_timing);
            }

            timing_info_push_start(copy_in_timing);
            cudaMemcpy(deviceObjects, reshaped_data, M * SUB_M * sizeof(double), cudaMemcpyHostToDevice);
            timing_info_push_end(copy_in_timing);
        
            timing_info_push_start(kernel_timing);
            find_nearest_cluster_block<<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, SUB_M, numClusters, i, deviceObjects, deviceClusters, deviceMembership, deviceIntermediates);
            cudaDeviceSynchronize(); checkLastCudaError();
            timing_info_push_end(kernel_timing);
        }

        timing_info_push_start(kernel_timing);
        compute_delta <<< 1, numReductionThreads, reductionBlockSharedDataSize >>>
        (deviceIntermediates, numClusterBlocks, numReductionThreads);
        cudaDeviceSynchronize(); checkLastCudaError();
        timing_info_push_end(kernel_timing);

        timing_info_push_start(copy_out_timing);
        int d;
        checkCuda(cudaMemcpy(&d, deviceIntermediates, sizeof(int), cudaMemcpyDeviceToHost));
        delta = (double)d;

        checkCuda(cudaMemcpy(membership, deviceMembership, numObjs*sizeof(int), cudaMemcpyDeviceToHost));
        timing_info_push_end(copy_out_timing);
        
        for (st = 0; st < (M / SUB_M); st++) {
            // TODO: fetch the data first. use r_thread here again?
            printf("i: %lu\n", st * SUB_M);
            // M = 65536, SUB_M = 1024
            for (sub_st = 0; sub_st < M / SUB_M; sub_st++) {
                timing_info_push_start(cluster_fetch_timing);
                sock_write_request(res->req_sock, id, sub_st, sub_st+1, SUB_M, 2, 0);
                sock_read_data(res->req_sock);
            
                resp = sock_read_offset(res->sock);
                if (resp == NULL) {
                    fprintf(stderr, "sync error before RDMA ops\n");
                    return clusters;
                }
                timing_info_push_end(cluster_fetch_timing);
                
                timing_info_push_start(cluster_reshape_timing);
                src_ptr = (double *) (res->buf + resp->offset);
                // pick 1024 * 1024 from 1024 * 65536
                col_reshape(reshaped_data + sub_st * SUB_M * SUB_M, src_ptr + st * SUB_M, M, SUB_M);
                if (sock_write_data(res->sock)) { /* just send a dummy char back and forth */
                    fprintf(stderr, "sync error before RDMA ops\n");
                    return clusters;
                }
                free(resp);
                timing_info_push_end(cluster_reshape_timing);
            }

            timing_info_push_start(cluster_assignment_timing);
            for (i = 0; i < SUB_M; i++) {
                /* find the array index of nestest cluster center */
                index = membership[st * SUB_M + i];
    
                /* update new cluster centers : sum of objects located within */
                newClusterSize[index]++;
                for (j=0; j<numCoords; j++) {
                    newClusters[j][index] += reshaped_data[j * SUB_M + i];
                }
            }
            timing_info_push_end(cluster_assignment_timing);
        }

        //  TODO: Flip the nesting order
        //  TODO: Change layout of newClusters to [numClusters][numCoords]
        /* average the sum and replace old cluster centers with newClusters */
        timing_info_push_start(cluster_assignment_timing);
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 0)
                    dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
                newClusters[j][i] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }
        delta /= numObjs;
        timing_info_push_end(cluster_assignment_timing);
    } while (delta > threshold && loop++ < 500);
    
    // send a signal to tell storage backend the iteration is done.
    sock_write_request(res->req_sock, -1, 0, 1, SUB_M, 0, 0);
    sock_read_data(res->req_sock);
    
    gettimeofday(&h_end, NULL);
    duration = ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
    
    printf("Kmeans End-to-end duration: %f ms\n", (float) duration / 1000);    
    printf("Col fetch time: %f ms\n", (float) timing_info_duration(col_fetch_timing) / 1000);
    printf("Reshape time: %f ms\n", (float) timing_info_duration(reshape_timing) / 1000);
    printf("Copy in time: %f ms\n", (float) timing_info_duration(copy_in_timing) / 1000);
    printf("Kernel time: %f ms\n", (float) timing_info_duration(kernel_timing) / 1000);
    printf("copy out time: %f ms\n", (float) timing_info_duration(copy_out_timing) / 1000);
    printf("cluster col fetch time: %f ms\n", (float) timing_info_duration(cluster_fetch_timing) / 1000);
    printf("cluster col reshape time: %f ms\n", (float) timing_info_duration(cluster_reshape_timing) / 1000);
    printf("cluster assignment time: %f ms\n", (float) timing_info_duration(cluster_assignment_timing) / 1000);

    struct timestamps *tss = NULL;
    FILE *fptr;
    tss = timing_info_get_timestamps(col_fetch_timing);
    fptr = fopen("col_fetch_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(col_fetch_timing);

    tss = timing_info_get_timestamps(reshape_timing);
    fptr = fopen("reshape_timing_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(reshape_timing);

    tss = timing_info_get_timestamps(copy_in_timing);
    fptr = fopen("copy_in_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(copy_in_timing);

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

    tss = timing_info_get_timestamps(cluster_fetch_timing);
    fptr = fopen("cluster_fetch_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(cluster_fetch_timing);    
    
    tss = timing_info_get_timestamps(cluster_reshape_timing);
    fptr = fopen("cluster_reshape_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(cluster_reshape_timing);   

    tss = timing_info_get_timestamps(cluster_assignment_timing);
    fptr = fopen("cluster_assignment_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(cluster_assignment_timing);

    *loop_iterations = loop + 1;
    printf("loop: %d\n", loop+1);
    /* allocate a 2D space for returning variable clusters[] (coordinates
    of cluster centers) */
    for (i = 0; i < numClusters; i++) {
        for (j = 0; j < numCoords; j++) {
            clusters[i][j] = dimClusters[j][i];
        }
    }

    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(deviceMembership));
    checkCuda(cudaFree(deviceIntermediates));
    free(reshaped_data);
    free(dimClusters[0]);
    free(dimClusters);
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);
    return clusters;
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
    uint64_t matrix_id, numClusters, numCoords, numObjs;
    int *membership;    /* [numObjs] */
    double **clusters;      /* [numClusters][numCoords] cluster center */
    double threshold;
    int loop_iterations;    
    
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

    // some default values
    threshold        = 0.001;
    // numClusters      = 4;

    if (argc < 6) {
        printf("usage: %s <table matrix id> <# of reference points> <# of attributes> <# of clusters> <port>\n", argv[0]);
        return EXIT_FAILURE;
    }

    matrix_id = (uint64_t) atoll(argv[1]);
    numObjs = (uint64_t) atoll(argv[2]);
    numCoords = (uint64_t) atoll(argv[3]);
    numClusters = (uint64_t) atoll(argv[4]);
    config.tcp_port = atoi(argv[5]);

    /* print the used parameters for info*/
    print_config(config);
    
    // initialize data for query points here.
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

    // start the timer for the core computation
    // membership: the cluster id for each data object
    membership = (int*) malloc(numObjs * sizeof(int));
    assert(membership != NULL);
    printf("calculating the result of pagerank\n");    

    clusters = nds_kmeans_block(&res, matrix_id, numCoords, numObjs, numClusters, SUB_M, threshold, membership, &loop_iterations);
    file_write(numClusters, numObjs, numCoords, clusters, membership);

    printf("\nPerforming **** Regular Kmeans (CUDA version) ****\n");
    printf("numObjs       = %lu\n", numObjs);
    printf("numCoords     = %lu\n", numCoords);
    printf("numClusters   = %lu\n", numClusters);
    printf("threshold     = %.4f\n", threshold);

    printf("Loop iterations    = %d\n", loop_iterations);
    
    close(res.sock);
    close(res.req_sock);
    munmap(hugepage_addr, BUF_SIZE);
    close(hugepage_fd);
    free(membership);
    free(clusters[0]);
    free(clusters);  
    return EXIT_SUCCESS;
}
