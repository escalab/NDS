/*********************************************************************************
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

The CUDA Kernel for Applying BFS on a loaded Graph. Created By Pawan Harish
**********************************************************************************/
#ifndef _KERNEL_H_
#define _KERNEL_H_

__global__ void
Kernel( uint64_t* g_graph_nodes, bool* g_graph_mask, bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int no_of_nodes) 
{
	uint64_t tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid < no_of_nodes && g_graph_mask[tid])
	{
		g_graph_mask[tid]=false;
		uint64_t offset = tid * no_of_nodes;
		uint64_t *node = g_graph_nodes + offset;
		for(int i = 0; i < no_of_nodes; i++)
		{
			if (node[i]) {
				if(!g_graph_visited[i])
				{
					g_cost[i]=g_cost[tid]+1;
					g_updating_graph_mask[i]=true;
				}
			}
		}
	}
}

#endif 
