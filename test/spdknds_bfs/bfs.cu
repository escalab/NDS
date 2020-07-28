/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <cuda.h>

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

// for timing
#include <sys/time.h>

extern "C" {
    #include "spdkrpc.h"
}

#define THREADS_PER_BLOCK 1024

__global__ void
Kernel2(bool* g_graph_mask, bool *g_updating_graph_mask, bool* g_graph_visited, bool *g_over, int no_of_nodes)
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < no_of_nodes && g_updating_graph_mask[tid])
	{
		g_graph_mask[tid] = true;
		g_graph_visited[tid] = true;
		*g_over = true;
		g_updating_graph_mask[tid] = false;
	}
}

__global__ void
Kernel(size_t st, uint64_t* g_graph_nodes, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int no_of_nodes, int num_of_subvertices) 
{
	uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t node_id = tid + st * num_of_subvertices;
	if(node_id < no_of_nodes && g_graph_mask[node_id])
	{
		g_graph_mask[node_id] = false;
		uint64_t offset = tid * no_of_nodes;
		uint64_t *node = g_graph_nodes + offset;
		for(int i = 0; i < no_of_nodes; i++)
		{
			if (node[i]) {
				if(!g_graph_visited[i])
				{
					g_cost[i] = g_cost[node_id] + 1;
					g_updating_graph_mask[i] = true;
				}
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
    size_t return_size, stripe_size, st, i;
    
    uint64_t *graph_h, *graph_d;
    struct JSONRPCClient client;
	int source = 0;

    // result
    int *cost_h, *cost_d;
	bool *graph_mask_h, *graph_mask_d, *updating_graph_mask_d, *graph_visited_d;

    // timing
    struct timeval h_start, h_end;
    uint64_t fetch_row_time = 0, kernel_1_time = 0, kernel_2_time = 0;

	bool *d_over;
	bool h_over;
	int k = 0;

    if (argc < 4) {
        printf("usage: %s <matrix id> <# of vertices> <# of subvertices>\n", argv[0]);
        exit(1);
    } 

    rc = connect_to_spdkrpc_server(&client);
    if (rc) {
        printf("cannot create conntection to SPDK RPC server");
        return rc;
    }

    graph_h = (uint64_t *) mmap_to_tensorstore_hugepage();
    if (graph_h == NULL) {
        return -1;
    }

    id = atoi(argv[1]);
    num_of_vertices = (size_t) atoi(argv[2]);
    num_of_subvertices = (size_t) atoi(argv[3]);
    
    stripe_size = num_of_vertices * num_of_subvertices * sizeof(int64_t);
	
	// subgraph initialization
    cudaMalloc((void **) &graph_d, stripe_size);

	// BFS initialization
	graph_mask_h = (bool *) calloc(num_of_vertices, sizeof(bool));
	cudaMalloc((void **) &graph_mask_d, sizeof(bool) * num_of_vertices);
	graph_mask_h[source] = true;
	cudaMemcpy(graph_mask_d, graph_mask_h, sizeof(bool) * num_of_vertices, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &updating_graph_mask_d, sizeof(bool) * num_of_vertices);
	cudaMemset(updating_graph_mask_d, 0, sizeof(bool) * num_of_vertices);

	bool *graph_visited_h = (bool *)calloc(sizeof(bool), num_of_vertices);
	graph_visited_h[source] = true;
	cudaMalloc((void **) &graph_visited_d, sizeof(bool) * num_of_vertices);
	cudaMemcpy(graph_visited_d, graph_visited_h, sizeof(bool) * num_of_vertices, cudaMemcpyHostToDevice);
	free(graph_visited_h);

	cost_h = (int *) malloc(sizeof(int) * num_of_vertices);
	cudaMalloc((void **) &cost_d, sizeof(int) * num_of_vertices);
	for (i = 0; i < num_of_vertices; i++) {
		cost_h[i] = -1;
	}	
	cost_h[source] = 0;
	cudaMemcpy(cost_d, cost_h, sizeof(int) * num_of_vertices, cudaMemcpyHostToDevice);

	//make a bool to check if the execution is over
	cudaMalloc((void**) &d_over, sizeof(bool));

	//Call the Kernel untill all the elements of Frontier are not false
	do
	{
		//if no thread changes this value then the loop stops
		h_over = false;
		cudaMemset(d_over, 0, sizeof(bool));
		for (st = 0; st < (num_of_vertices / num_of_subvertices); st++) {
			// check whether we need to fetch the graph for the subsection
			for (i = st*num_of_subvertices; i < (st+1) * num_of_subvertices; i++) {
				if (graph_mask_h[i]) {
					gettimeofday(&h_start, NULL);
					return_size = tensorstore_get_row_stripe_submatrix(&client, id, st, st+1, num_of_subvertices);					
					cudaMemcpy(graph_d, graph_h, return_size, cudaMemcpyHostToDevice);
					gettimeofday(&h_end, NULL);
					fetch_row_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   		
					break;
				}
			}

			// if we don't need to fetch subgraph for this subsection, we can skip.
			if (i == ((st+1) * num_of_subvertices)) {
				continue;
			}

			// printf("kernel1\n");
			// updating the cost of nodes that are masked in d_graph_mask
			gettimeofday(&h_start, NULL);
			Kernel<<<(num_of_subvertices+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(st, graph_d, graph_mask_d, updating_graph_mask_d, graph_visited_d, cost_d, num_of_vertices, num_of_subvertices);
            gettimeofday(&h_end, NULL);
            kernel_1_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   
		}

		// printf("kernel2\n");
		// expanding the search to the next step
		gettimeofday(&h_start, NULL);
		Kernel2<<<(num_of_vertices+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(graph_mask_d, updating_graph_mask_d, graph_visited_d, d_over, num_of_vertices);
		// check if kernel execution generated and error
		gettimeofday(&h_end, NULL);
		kernel_2_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   
		
		cudaMemcpy(&h_over, d_over, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaMemcpy(graph_mask_h, graph_mask_d, sizeof(bool) * num_of_vertices, cudaMemcpyDeviceToHost);
		printf("h_over %d\n", h_over);
		k++;
	}
	while(h_over);

	cudaMemcpy(cost_h, cost_d, sizeof(int) * num_of_vertices, cudaMemcpyDeviceToHost) ;

    FILE *fp = fopen("log.txt", "w");
    for (i = 0; i < num_of_vertices; i++) {
        fprintf(fp, "%d) cost:%d\n", i, cost_h[i]);
    }

    printf("row fetch time: %f ms\n", (float) fetch_row_time / 1000);
    printf("kernel 1 time : %f ms\n", (float) kernel_1_time / 1000);
    printf("kernel 2 time: %f ms\n", (float) kernel_2_time / 1000);

    fclose(fp);
	// cleanup
	free(cost_h);
	free(graph_mask_h);
    munmap(graph_h, HUGEPAGE_SZ);
    cudaFree(graph_d);
    cudaFree(graph_mask_d);
    cudaFree(updating_graph_mask_d);
    cudaFree(graph_visited_d);
    cudaFree(d_over);

    return 0;
}