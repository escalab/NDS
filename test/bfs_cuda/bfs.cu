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

#define MAX_THREADS_PER_BLOCK 512

int num_nodes;
int edge_list_size;
FILE *fp;

#include "kernel.cu"
#include "kernel2.cu"

void BFSGraph(uint64_t *graph, int source, int num_nodes);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    uint64_t *graph;
    size_t size;
    int fd, num_nodes;

    if (argc < 3) {
        printf("usage: %s <graph path> <num of nodes>\n", argv[0]);
        exit(1);
    }
    fd = open(argv[1], O_RDONLY);
    num_nodes = atoi(argv[2]);

    size = sizeof(uint64_t) * num_nodes;
    size *= num_nodes;
    graph = (uint64_t *) mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);

    BFSGraph(graph, 0, num_nodes);
    close(fd);
    munmap(graph, size);
    return 0;
}

void Usage(int argc, char** argv){
	fprintf(stderr,"Usage: %s <graph binary> <num node>\n", argv[0]);
}
////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph(uint64_t *graph, int source, int num_nodes) 
{
	int num_of_blocks = 1;
	int num_of_threads_per_block = num_nodes;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(num_nodes>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(num_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}

	// allocate host memory
	uint64_t size;
    uint64_t *d_graph_nodes;
	
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*num_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*num_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*num_nodes);

	size = sizeof(uint64_t) * num_nodes;
	size *= num_nodes;
	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	//Copy the Node list to device memory
	cudaMalloc( (void**) &d_graph_nodes, size) ;
	cudaMemcpy( d_graph_nodes, graph, size, cudaMemcpyHostToDevice) ;

	//Copy the Mask to device memory
	bool* d_graph_mask;
	cudaMalloc( (void**) &d_graph_mask, sizeof(bool)*num_nodes) ;
	cudaMemcpy( d_graph_mask, h_graph_mask, sizeof(bool)*num_nodes, cudaMemcpyHostToDevice) ;

	bool* d_updating_graph_mask;
	cudaMalloc( (void**) &d_updating_graph_mask, sizeof(bool)*num_nodes) ;
	cudaMemcpy( d_updating_graph_mask, h_updating_graph_mask, sizeof(bool)*num_nodes, cudaMemcpyHostToDevice) ;

	//Copy the Visited nodes array to device memory
	bool* d_graph_visited;
	cudaMalloc( (void**) &d_graph_visited, sizeof(bool)*num_nodes) ;
	cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(bool)*num_nodes, cudaMemcpyHostToDevice) ;

	// allocate mem for the result on host side
	int *h_cost = (int*) malloc( sizeof(int)*num_nodes);
	for (int i=0;i<num_nodes;i++) {
		h_cost[i] = -1;
	}
	h_cost[source] = 0;
	
	// allocate device memory for result
	int* d_cost;
	cudaMalloc( (void**) &d_cost, sizeof(int)*num_nodes);
	cudaMemcpy( d_cost, h_cost, sizeof(int)*num_nodes, cudaMemcpyHostToDevice) ;

	//make a bool to check if the execution is over
	bool *d_over;
	cudaMalloc( (void**) &d_over, sizeof(bool));

	printf("Copied Everything to GPU memory\n");

	// setup execution parameters
	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

	int k=0;
	printf("Start traversing the tree\n");
	bool stop;
	//Call the Kernel untill all the elements of Frontier are not false
	do
	{
		//if no thread changes this value then the loop stops
		stop=false;
		cudaMemcpy( d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) ;

		// updating the cost of nodes that are masked in d_graph_mask
		Kernel<<< grid, threads, 0 >>>( d_graph_nodes, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, num_nodes);
		// check if kernel execution generated and error
		
		// expanding the search to the next step
		Kernel2<<< grid, threads, 0 >>>( d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, num_nodes);
		// check if kernel execution generated and error
		

		cudaMemcpy( &stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost) ;
		k++;
	}
	while(stop);


	printf("Kernel Executed %d times\n",k);

	// copy result from device to host
	cudaMemcpy( h_cost, d_cost, sizeof(int)*num_nodes, cudaMemcpyDeviceToHost) ;

	//Store the result into a file
	FILE *fpo = fopen("result.txt","w");
	for(int i=0;i<num_nodes;i++)
		fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");


	// cleanup memory
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
	cudaFree(d_graph_nodes);
	cudaFree(d_graph_mask);
	cudaFree(d_updating_graph_mask);
	cudaFree(d_graph_visited);
	cudaFree(d_cost);
}
