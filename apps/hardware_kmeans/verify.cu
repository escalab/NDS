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
        ptr_a = conf->deviceObjects + dsize * (count % IO_QUEUE_SZ);

        cudaMemcpyFromMmap(conf, (char *) ptr_a, (char *) conf->hugepage_addr, dsize * sizeof(double), conf->col_fetch_timing);
        count++;

        entry->deviceObjects = ptr_a;
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
    return NULL;
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
    
    struct response *response = NULL;
    double *buf;
    /* pick first numClusters elements of objects[] as initial cluster centers*/
    malloc2D(clusters, numClusters, numCoords, double);
    malloc2D(dimClusters, numCoords, numClusters, double);

    sock_write_request(res->req_sock, id, 0, 1, SUB_M, 3, 0);
    sock_read_data(res->req_sock);

    response = sock_read_offset(res->sock);
    if (response == NULL) {
        fprintf(stderr, "sync error before RDMA ops\n");
        return clusters;
    }
    buf = (double *) (res->buf + response->offset);
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = buf[i * SUB_M + j];
        }
    }
    free(response);
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

    struct fifo *sending_queue;
    struct fifo *complete_queue; 
    struct fifo_entry *entries = (struct fifo_entry *) calloc(IO_QUEUE_SZ, sizeof(struct fifo_entry));
    struct fifo_entry *entry = NULL;

    struct timing_info *queue_timing;
    struct timing_info *col_fetch_timing;
    struct timing_info *copy_in_timing;
    struct timing_info *kernel_timing;
    struct timing_info *copy_out_timing;
    struct timing_info *cluster_fetch_timing;    
    struct timing_info *cluster_assignment_timing;    

    pthread_t f_thread_id; 
    struct fetch_conf f_conf;

    pthread_t r_thread_id; 
    struct request_conf r_conf;

    struct timeval h_start, h_end;
    long duration;

    // initialization
    // tested beforehand and found out we will have 11 loops;
    size_t total_iteration = (M / SUB_M) * 32;
    queue_timing = timing_info_new(total_iteration);
    if (queue_timing == NULL) {
        printf("cannot create queue_timing\n");
        return clusters;
    }

    col_fetch_timing = timing_info_new(total_iteration);
    if (col_fetch_timing == NULL) {
        printf("cannot create col_fetch_timing\n");
        return clusters;
    }

    copy_in_timing = timing_info_new(total_iteration * 2);
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

    cluster_fetch_timing = timing_info_new(total_iteration);
    if (cluster_fetch_timing == NULL) {
        printf("cannot create cluster_fetch_timing\n");
        return clusters;
    }

    cluster_assignment_timing = timing_info_new(total_iteration);
    if (cluster_assignment_timing == NULL) {
        printf("cannot create cluster_assignment_timing\n");
        return clusters;
    }

    // it causes problem if size == 1
    sending_queue = fifo_new(IO_QUEUE_SZ * 2);
	if (sending_queue == NULL) {
        printf("cannot create sending_queue\n");
        return clusters;
    }
    
    complete_queue = fifo_new(IO_QUEUE_SZ * 2);
	if (complete_queue == NULL) {
        printf("cannot create complete_queue\n");
        return clusters;
    }
    
    for (i = 0; i < IO_QUEUE_SZ; i++) {
        fifo_push(complete_queue, entries + i);
    }


    // create thread here
    f_conf.res = res;
    f_conf.m = M;
    f_conf.sub_m = SUB_M;
    f_conf.deviceObjects = deviceObjects;
    f_conf.hugepage_addr = res->buf;
    f_conf.sending_queue = sending_queue;
    f_conf.complete_queue = complete_queue;
    f_conf.col_fetch_timing = col_fetch_timing;
    f_conf.copy_in_timing = copy_in_timing;
    r_conf.res = res;
    r_conf.id = id;
    r_conf.sub_m = SUB_M;

    timing_info_set_starting_time(queue_timing);
    timing_info_set_starting_time(col_fetch_timing);
    timing_info_set_starting_time(copy_in_timing);
    timing_info_set_starting_time(kernel_timing);
    timing_info_set_starting_time(copy_out_timing);
    timing_info_set_starting_time(cluster_fetch_timing);
    timing_info_set_starting_time(cluster_assignment_timing);

    // computation part
    gettimeofday(&h_start, NULL);
    do {
        pthread_create(&r_thread_id, NULL, request_thread, &r_conf); 
        pthread_create(&f_thread_id, NULL, fetch_thread, &f_conf);
        checkCuda(cudaMemset(deviceIntermediates, 0, numReductionThreads*sizeof(unsigned int)));
        checkCuda(cudaMemcpy(deviceClusters, dimClusters[0], numClusters*numCoords*sizeof(double), cudaMemcpyHostToDevice));
        
        for (i = 0; i < numObjs; i += SUB_M) {
            // printf("i: %d\n", i);
            // TODO: need cudaMemcpy2D
            timing_info_push_start(queue_timing);
            entry = (struct fifo_entry *) fifo_pop(sending_queue);
            timing_info_push_end(queue_timing);

            timing_info_push_start(kernel_timing);
            find_nearest_cluster_block<<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, SUB_M, numClusters, i, entry->deviceObjects, deviceClusters, deviceMembership, deviceIntermediates);
            cudaDeviceSynchronize(); checkLastCudaError();
            fifo_push(complete_queue, entry);
            timing_info_push_end(kernel_timing);

        }
        pthread_join(r_thread_id, NULL); 

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
        
        uint64_t st;
        pthread_create(&r_thread_id, NULL, request_thread, &r_conf); 
        for (st = 0; st < (M / SUB_M); st++) {
            // TODO: fetch the data first. use r_thread here again?
            // printf("i: %lu\n", st * SUB_M);
            timing_info_push_start(cluster_fetch_timing);
            response = sock_read_offset(res->sock);
            if (response == NULL) {
                fprintf(stderr, "sync error before RDMA ops\n");
                return clusters;
            }
            buf = (double *) (res->buf + response->offset);
            timing_info_push_end(cluster_fetch_timing);

            timing_info_push_start(cluster_assignment_timing);
            for (i = 0; i < SUB_M; i++) {
                /* find the array index of nestest cluster center */
                index = membership[st * SUB_M + i];
    
                /* update new cluster centers : sum of objects located within */
                newClusterSize[index]++;
                for (j=0; j<numCoords; j++) {
                    newClusters[j][index] += buf[j * SUB_M + i];
                }
            }
            timing_info_push_end(cluster_assignment_timing);

            if (sock_write_data(res->sock)) { /* just send a dummy char back and forth */
                fprintf(stderr, "sync error before RDMA ops\n");
                return clusters;
            }
        }
        pthread_join(r_thread_id, NULL); 

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
    printf("Copy in time: %f ms\n", (float) timing_info_duration(copy_in_timing) / 1000);
    printf("sending_queue waiting time: %f ms\n", (float) timing_info_duration(queue_timing) / 1000);
    printf("Kernel time: %f ms\n", (float) timing_info_duration(kernel_timing) / 1000);
    printf("copy out time: %f ms\n", (float) timing_info_duration(copy_out_timing) / 1000);
    printf("cluster col fetch time: %f ms\n", (float) timing_info_duration(cluster_fetch_timing) / 1000);
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

    tss = timing_info_get_timestamps(cluster_fetch_timing);
    fptr = fopen("cluster_fetch_ts.bin", "wb");
    fwrite(&tss->count, sizeof(uint64_t), 1, fptr);
    fwrite(tss->timestamps, sizeof(uint64_t), tss->count * 2, fptr);
    fclose(fptr);
    timing_info_free_timestamps(tss);    
    timing_info_free(cluster_fetch_timing);    
    
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

    free(entries);
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

    // start the timer for the core computation
    // membership: the cluster id for each data object
    membership = (int*) malloc(numObjs * sizeof(int));
    assert(membership != NULL);
    printf("calculating the result of pagerank\n");    

    clusters = nds_kmeans_block(&res, matrix_id, numCoords, numObjs, numClusters, SUB_M, threshold, membership, &loop_iterations);
    file_write(numClusters, numObjs, numCoords, clusters, membership);

    if (resources_destroy(&res)) {
        fprintf(stderr, "failed to destroy resources\n");
        exit(1);
    }
    printf("\nPerforming **** Regular Kmeans (CUDA version) ****\n");
    printf("numObjs       = %lu\n", numObjs);
    printf("numCoords     = %lu\n", numCoords);
    printf("numClusters   = %lu\n", numClusters);
    printf("threshold     = %.4f\n", threshold);

    printf("Loop iterations    = %d\n", loop_iterations);
    
    // close(res.sock);
    close(res.req_sock);
    munmap(hugepage_addr, BUF_SIZE);
    close(hugepage_fd);
    free(membership);
    free(clusters[0]);
    free(clusters);  
    return EXIT_SUCCESS;
}
