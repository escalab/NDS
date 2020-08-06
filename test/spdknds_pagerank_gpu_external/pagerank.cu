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

#define RANDOMRESETPROB 0.15
#define THREADS_PER_BLOCK 1024

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)


void get_outedges(double* graph, double* outedges, size_t interval_st, size_t num_of_vertices, size_t num_of_subvertices) {
    checkKernelErrors(cudaMemcpy(outedges, (graph + interval_st * num_of_vertices), sizeof(double) * num_of_vertices * num_of_subvertices, cudaMemcpyHostToDevice));
}

void put_outedges(double* graph, double* outedges, size_t interval_st, size_t num_of_vertices, size_t num_of_subvertices) {
    checkKernelErrors(cudaMemcpy((graph + interval_st * num_of_vertices), outedges, sizeof(double) * num_of_vertices * num_of_subvertices, cudaMemcpyDeviceToHost));
    msync((graph + interval_st * num_of_vertices), sizeof(double) * num_of_vertices * num_of_subvertices, MS_SYNC);
}

void get_inedges(double* graph, double* inedges, size_t interval_st, size_t num_of_vertices, size_t num_of_subvertices) {
    size_t i;
    double *graph_ptr, *inedges_ptr;
    graph_ptr = graph + interval_st;
    inedges_ptr = inedges;
    // in column favor
    for (i = 0; i < num_of_vertices; i++) {
        checkKernelErrors(cudaMemcpy(inedges_ptr, graph_ptr, sizeof(double) * num_of_subvertices, cudaMemcpyHostToDevice));
        graph_ptr += num_of_vertices;
        inedges_ptr += num_of_subvertices;
    }
}

// construct vertices metadata first
__global__ void pagerank_update(double *vertices, size_t st, double *in_graph, double *out_graph, int *updated, size_t num_of_vertices, size_t num_of_subvertices, int iter, int niters) {
    // v.outc is num_outedges()
    // needs: v.num_inedges(), v.inedge(), v.id(), v.outc, v.set_data
    size_t id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= num_of_subvertices) {
        return;
    }
    size_t i, outc = 0;
    double *inedge, *outedge;
    double sum = 0;

    inedge = in_graph + id;
    outedge = out_graph + id * num_of_vertices;
    id = st + id;

    for (i = 0; i < num_of_vertices; i++) {
        if (i != id && outedge[i] != 0) {
            outc++;
        }
    }

    if (iter == 0) {
        for (i = 0; i < num_of_vertices; i++) {
            if (i != id && outedge[i] != 0) {
                outedge[i] = 1.0 / outc;
                *updated = 1;
            }
        }
        vertices[id] = RANDOMRESETPROB;
    } else {
        for (i = 0; i < num_of_vertices; i++) {
            // we don't consider self-loop
            if (inedge[i * num_of_subvertices] && i != id) {
                sum += inedge[i * num_of_subvertices];
            }
        }
        double pagerank = (RANDOMRESETPROB + (1 - RANDOMRESETPROB) * sum);
        if (outc > 0) {
            double pagerankcont = pagerank / (double) outc;
            for (i = 0; i < num_of_vertices; i++) {
                if (i != id && outedge[i] != 0) {
                    outedge[i] = pagerankcont;
                    *updated = 1;
                }
            }
        } 
        vertices[id] = pagerank;
    }
}

/**
 * TODO section:
 * maybe we can borrow the log system from graphchi?
 */
 int main(int argc, char** argv) {
    double *graph, *outedges, *inedges;
    int rc, temp, fd_in, fd_out;
    size_t num_of_vertices, num_of_subvertices, niters;
    size_t return_size, stripe_size, iter, st, i;
    int updated_h, *updated_d;
    struct JSONRPCClient client;

    // result
    double *vertices, *vertices_d;

    // timing
    struct timeval h_start, h_end;
    uint64_t fetch_row_time = 0, fetch_col_time = 0, write_row_time = 0, kernel_time = 0;

    if (argc < 6) {
        printf("usage: %s <graph 1 path> <graph 2 path> <# of vertices> <# of subvertices> <niters>\n", argv[0]);
        exit(1);
    } 

    // I/O part, open in mmap mode
    fd_in = atoi(argv[1]);
    fd_out = atoi(argv[2]);
    num_of_vertices = (size_t) atoi(argv[3]);
    num_of_subvertices = (size_t) atoi(argv[4]);
    niters = (size_t) atoi(argv[5]);

    stripe_size = num_of_vertices * num_of_subvertices * sizeof(double);
    printf("matrix 1: %d\n", fd_in);
    printf("matrix 2: %d\n", fd_out);
    printf("num_of_vertices: %lu\n", num_of_vertices);
    printf("num_of_subvertices: %lu\n", num_of_subvertices);
    printf("niters: %lu\n", niters);

    rc = connect_to_spdkrpc_server(&client);
    if (rc) {
        printf("cannot create conntection to SPDK RPC server");
        return rc;
    }

    graph = (double *) mmap_to_tensorstore_hugepage();
    if (graph == NULL) {
        return -1;
    }

    // PR initialization
    vertices = (double *) calloc(sizeof(double), num_of_vertices);

    // for (i = 0; i < num_of_vertices; i++) {
    //     vertices[i] = RANDOMRESETPROB;
    // }

    cudaMalloc((void **) &vertices_d, sizeof(double) * num_of_vertices);
    cudaMemset(vertices_d, 0, sizeof(double) * num_of_vertices);
    // cudaMemcpy(vertices_d, vertices, sizeof(double) * num_of_vertices, cudaMemcpyHostToDevice);

    // subgraph initialization
    checkKernelErrors(cudaMalloc((void **) &outedges, stripe_size));
    checkKernelErrors(cudaMalloc((void **) &inedges, stripe_size));
    
    checkKernelErrors(cudaMalloc((void **) &updated_d, sizeof(int)));

    // Kernel loop (inspired by GraphChi engine)
    for (iter = 0; iter < niters; iter++) {
        // userprogram.before_iteration(iter, chicontext);
        // do nothing in example

        // scheduler function
        // do nothing in example

        // shuffle function
        // do nothing in example

        // Interval loop
        // assume we have no subinterval in an interval.
        for (st = 0; st < (num_of_vertices / num_of_subvertices); st++) {
            // checkKernelErrors(cudaMemcpy(updated_d, &updated_h, sizeof(int), cudaMemcpyHostToDevice));
            updated_h = 0;
            checkKernelErrors(cudaMemset(updated_d, 0, sizeof(int)));
            // cudaMemset(outedges, 0, stripe_size);

            printf("st: %lu\n", st * num_of_subvertices);
            /* preprocessing */

            // userprogram.before_exec_interval(interval_st, interval_en, chicontext);
            // do nothing in example

            // flush things back from sliding_shards

            // create a new memory shard
            gettimeofday(&h_start, NULL);
            return_size = tensorstore_get_row_stripe_submatrix(&client, fd_out, st, st+1, num_of_subvertices);
            cudaMemcpy(outedges, graph, return_size, cudaMemcpyHostToDevice);
            gettimeofday(&h_end, NULL);
            fetch_row_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   

            gettimeofday(&h_start, NULL);
            return_size = tensorstore_get_col_stripe_submatrix(&client, fd_in, st, st+1, num_of_subvertices);
            cudaMemcpy(inedges, graph, return_size, cudaMemcpyHostToDevice);
            gettimeofday(&h_end, NULL);
            fetch_col_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);   

            // printf("%p %p\n", in_graph, out_graph);
            // printf("%p %p\n", outedges, inedges);

            // initialize vertices & edge data

            // load data
            // load_before_updates(vertices);

            /* execution part */

            // exec_updates(userprogram, vertices);
            // exec_updates(GraphChiProgram<VertexDataType, EdgeDataType, svertex_t> &userprogram, std::vector<svertex_t> &vertices)
            
            // update vertices one by one
            // userprogram.update(v, chicontext);
            // update(graphchi_vertex<VertexDataType, EdgeDataType> &v, graphchi_context &ginfo);
            gettimeofday(&h_start, NULL);
            pagerank_update<<<(num_of_subvertices+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(vertices_d, st * num_of_subvertices, inedges, outedges, updated_d, num_of_vertices, num_of_subvertices, iter, niters);
            gettimeofday(&h_end, NULL);
            kernel_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);                                                    
            
            cudaDeviceSynchronize();
            /* postprocessing */
            // printf("%p %p\n", &updated_h, updated_d);
            checkKernelErrors(cudaMemcpy(&updated_h, updated_d, sizeof(int), cudaMemcpyDeviceToHost));
            
            if (updated_h) {
                gettimeofday(&h_start, NULL);
                cudaMemcpy(graph, outedges, return_size, cudaMemcpyDeviceToHost);
                return_size = tensorstore_write_row_stripe_submatrix(&client, fd_out, st, st+1, num_of_subvertices);
                gettimeofday(&h_end, NULL);
                write_row_time += ((h_end.tv_sec - h_start.tv_sec) * 1000000) + (h_end.tv_usec - h_start.tv_usec);                                                        
            }

            // save_vertices(vertices);
            // nothing is modified in PR application

            // memoryshard->commit(modifies_inedges, modifies_outedges & !disable_outedges);
            // memoryshard->commit(0, 0) in PR. doesn't need to commit back

            // userprogram.after_exec_interval(interval_st, interval_en, chicontext);
            // do nothing in PR
        }
        
        // swap
        temp = fd_in;
        fd_in = fd_out;
        fd_out = temp;

        // userprogram.after_iteration(iter, chicontext);
        // do nothing in PR

        // flush all stuff in sliding_shards
        // nothing inside maybe. it is just a buffer for updating
    }
    cudaMemcpy(vertices, vertices_d, sizeof(double) * num_of_vertices, cudaMemcpyDeviceToHost);

    FILE *fp = fopen("log.txt", "w");
    for (i = 0; i < num_of_vertices; i++) {
        fprintf(fp, "%lu %f\n", i, vertices[i]);
    }
    fclose(fp);

    printf("row fetch time: %f ms\n", (float) fetch_row_time / 1000);
    printf("col fetch time: %f ms\n", (float) fetch_col_time / 1000);
    printf("kernel time: %f ms\n", (float) kernel_time / 1000);
    printf("row write time: %f ms\n", (float) write_row_time / 1000);

    // cleanup
    cudaFree(outedges);
    cudaFree(inedges);
    cudaFree(vertices_d);
    cudaFree(updated_d);
    free(vertices);

    return 0;
}