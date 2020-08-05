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
    double *in_graph, *out_graph, *temp, *outedges, *inedges;
    int fd_1, fd_2;
    size_t num_of_vertices, num_of_subvertices, niters;
    size_t return_size, stripe_size, iter, st, i;
    int updated_h, *updated_d;
    // result
    double *vertices, *vertices_d;

    // timing
    struct timeval h_start, h_end;
    long duration;

    if (argc < 6) {
        printf("usage: %s <graph 1 path> <graph 2 path> <# of vertices> <# of subvertices> <niters>\n", argv[0]);
        exit(1);
    } 

    // I/O part, open in mmap mode
    fd_1 = open(argv[1], O_RDWR);
    fd_2 = open(argv[2], O_RDWR);
    num_of_vertices = (size_t) atoi(argv[3]);
    num_of_subvertices = (size_t) atoi(argv[4]);
    niters = (size_t) atoi(argv[5]);
    in_graph = (double *) mmap(NULL, sizeof(double) * num_of_vertices * num_of_vertices, PROT_READ | PROT_WRITE, MAP_SHARED, fd_1, 0);
    out_graph = (double *) mmap(NULL, sizeof(double) * num_of_vertices * num_of_vertices, PROT_READ | PROT_WRITE, MAP_SHARED, fd_2, 0);

    stripe_size = num_of_vertices * num_of_subvertices * sizeof(double);
    printf("path 1: %s\n", argv[1]);
    printf("path 2: %s\n", argv[2]);
    printf("num_of_vertices: %lu\n", num_of_vertices);
    printf("num_of_subvertices: %lu\n", num_of_subvertices);
    printf("niters: %lu\n", niters);

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
        for (st = 0; st < num_of_vertices; st += num_of_subvertices) {
            // checkKernelErrors(cudaMemcpy(updated_d, &updated_h, sizeof(int), cudaMemcpyHostToDevice));
            updated_h = 0;
            checkKernelErrors(cudaMemset(updated_d, 0, sizeof(int)));
            // cudaMemset(outedges, 0, stripe_size);

            printf("st: %lu\n", st);
            /* preprocessing */

            // userprogram.before_exec_interval(interval_st, interval_en, chicontext);
            // do nothing in example

            // flush things back from sliding_shards

            // create a new memory shard
            get_outedges(out_graph, outedges, st, num_of_vertices, num_of_subvertices);
            get_inedges(in_graph, inedges, st, num_of_vertices, num_of_subvertices);
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
            pagerank_update<<<(num_of_subvertices+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(vertices_d, st, inedges, outedges, updated_d, num_of_vertices, num_of_subvertices, iter, niters);
            cudaError_t __err = cudaGetLastError();                 
            if (__err != cudaSuccess) {                             
              printf("Line %d: failed: %s\n", __LINE__, cudaGetErrorString(__err));                    
              abort();                                              
            }                                                       
            
            cudaDeviceSynchronize();
            /* postprocessing */
            // printf("%p %p\n", &updated_h, updated_d);
            checkKernelErrors(cudaMemcpy(&updated_h, updated_d, sizeof(int), cudaMemcpyDeviceToHost));
            
            if (updated_h) {
                printf("updating\n");
                put_outedges(out_graph, outedges, st, num_of_vertices, num_of_subvertices);
            }

            // save_vertices(vertices);
            // nothing is modified in PR application

            // memoryshard->commit(modifies_inedges, modifies_outedges & !disable_outedges);
            // memoryshard->commit(0, 0) in PR. doesn't need to commit back

            // userprogram.after_exec_interval(interval_st, interval_en, chicontext);
            // do nothing in PR
        }
        
        // swap
        temp = in_graph;
        in_graph = out_graph;
        out_graph = temp;

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
    // cleanup
    munmap(in_graph, sizeof(double) * num_of_vertices * num_of_vertices);
    close(fd_1);

    munmap(out_graph, sizeof(double) * num_of_vertices * num_of_vertices);
    close(fd_2);

    cudaFree(outedges);
    cudaFree(inedges);
    cudaFree(vertices_d);
    cudaFree(updated_d);
    free(vertices);

    return 0;
}