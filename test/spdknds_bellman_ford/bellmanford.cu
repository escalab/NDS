#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// for timing
#include <sys/time.h>

extern "C" {
    #include "spdkrpc.h"
}

#define INF INT_MAX
#define THREADS_PER_BLOCK 1024

/*
 * This is a CHECK function to check CUDA calls
 */
 #define CHECK(call)                                                            \
 {                                                                              \
     const cudaError_t error = call;                                            \
     if (error != cudaSuccess)                                                  \
     {                                                                          \
         fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
         fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                 cudaGetErrorString(error));                                    \
                 exit(1);                                                       \
     }                                                                          \
 }

 
__global__ void bellman_ford_one_iter(size_t n, size_t sub_n, size_t st, int64_t *d_subgraph, int64_t *d_dist, bool *d_has_next){
	size_t v = blockDim.x * blockIdx.x + threadIdx.x;
	size_t u;
	int64_t weight, new_dist;
	int64_t *node;
	
	if (v > sub_n) {
		return;
	}
	
	node = d_subgraph + v;
	v = v + st;
	for (u = 0; u < n; u++){
		weight = node[u * sub_n]; // row is src, col is dst
		if (weight > 0) {
			new_dist = d_dist[u] + weight;
			if(new_dist < d_dist[v]){
				d_dist[v] = new_dist;
				*d_has_next = true;
			}
		}
	}
}

/**
 * TODO section:
 * maybe we can borrow the log system from graphchi?
 */
int main(int argc, char** argv) {
    int id, rc;
    size_t num_of_vertices, num_of_subvertices;
    size_t return_size, stripe_sz, iter_num, st, i;
    
    int64_t *subgraph, *d_subgraph;
	bool *d_has_next, h_has_next;

    struct JSONRPCClient client;

    // result
    int64_t *dist, *d_dist;

    // timing
    struct timeval h_start, h_end;
    uint64_t fetch_col_time = 0, kernel_time = 0;

    if (argc < 4) {
        printf("usage: %s <matrix id> <# of vertices> <# of subvertices>\n", argv[0]);
        exit(1);
    } 

    rc = connect_to_spdkrpc_server(&client);
    if (rc) {
        printf("cannot create conntection to SPDK RPC server");
        return rc;
    }

    subgraph = (int64_t *) mmap_to_tensorstore_hugepage();
    if (subgraph == NULL) {
        return -1;
    }

    id = atoi(argv[1]);
    num_of_vertices = (size_t) atoi(argv[2]);
    num_of_subvertices = (size_t) atoi(argv[3]);
    
    stripe_sz = num_of_vertices * num_of_subvertices * sizeof(int64_t);

	cudaMalloc(&d_subgraph, stripe_sz);
	cudaMalloc(&d_dist, sizeof(int64_t) * num_of_vertices);
	cudaMalloc(&d_has_next, sizeof(bool));

    bool has_negative_cycle = false;
    
    dist = (int64_t *) calloc(sizeof(int64_t), num_of_vertices);
	for(i = 0 ; i < num_of_vertices; i++){
		dist[i] = INF;
	}

	dist[0] = 0;
	cudaMemcpy(d_dist, dist, sizeof(int64_t) * num_of_vertices, cudaMemcpyHostToDevice);

    iter_num = 0;
	do {
		h_has_next = false;
		cudaMemcpy(d_has_next, &h_has_next, sizeof(bool), cudaMemcpyHostToDevice);
        for (st = 0; st < (num_of_vertices / num_of_subvertices); st++) {
            gettimeofday(&h_start, NULL);
            return_size = tensorstore_get_col_stripe_submatrix(&client, id, st, st+1, num_of_subvertices);
            cudaMemcpy(d_subgraph, subgraph, return_size, cudaMemcpyHostToDevice);
            gettimeofday(&h_end, NULL);
            fetch_col_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   

            gettimeofday(&h_start, NULL);
			bellman_ford_one_iter<<<(num_of_subvertices+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(num_of_vertices, num_of_subvertices, st * num_of_subvertices, d_subgraph, d_dist, d_has_next);
            CHECK(cudaDeviceSynchronize());
            gettimeofday(&h_end, NULL);
            kernel_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);
		}
		cudaMemcpy(&h_has_next, d_has_next, sizeof(bool), cudaMemcpyDeviceToHost);

		iter_num++;

		if (iter_num >= num_of_vertices - 1){
			has_negative_cycle = true;
			break;
		}
	} while (h_has_next);

    
    FILE *fp = fopen("log.txt", "w");
	if (!has_negative_cycle){
        cudaMemcpy(dist, d_dist, sizeof(int64_t) * num_of_vertices, cudaMemcpyDeviceToHost);
        for (i = 0; i < num_of_vertices; i++) {
			if (dist[i] > INF) {
				dist[i] = INF;
			}
			fprintf(fp, "%lu %lu\n", i, dist[i]);
        }
	} else {
		fprintf(fp, "FOUND NEGATIVE CYCLE!\n");
	}
    printf("iteration time: %lu\n", iter_num);
    printf("col fetch time: %f ms\n", (float) fetch_col_time / 1000);
    printf("kernel time: %f ms\n", (float) kernel_time / 1000);

    fclose(fp);
    
    // cleanup
    free(dist);
    cudaFree(d_subgraph);
	cudaFree(d_dist);
    cudaFree(d_has_next);

    return 0;
}