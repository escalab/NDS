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

#include "spdkrpc.h"

#define RANDOMRESETPROB 0.15

// void get_outedges(int64_t* graph, int64_t* outedges, size_t interval_st, size_t num_of_vertices_per_stripe, size_t num_of_vertices) {
//     memcpy(outedges, (graph + interval_st * num_of_vertices), STRIPE_SZ);
// }

// void get_inedges(int64_t* graph, int64_t* inedges, size_t interval_st, size_t interval_en, size_t num_of_vertices) {
//     size_t i, j, row;
//     int64_t *graph_ptr, *inedges_ptr;
//     // in column favor but transpose into row.
//     for (i = 0; i < num_of_vertices; i++) {
//         for (row = 0, j = interval_st; j < interval_en; row++, j++) {
//             // printf("i: %lu, j: %lu, row: %lu\n", i, j, row);
//             *(inedges + row * num_of_vertices + i) = *(graph + i * num_of_vertices + j);
//         }
//     }
// }

// construct vertices metadata first

void pagerank_update(double* prev_pr, double* curr_pr, double *vertices, int id, int64_t* inedge, int64_t *outedge, int num_of_vertices, int iter, int niters) {
    // v.outc is num_outedges()
    // needs: v.num_inedges(), v.inedge(), v.id(), v.outc, v.set_data
    int i, outc = 0;
    double sum = 0;

    for (i = 0; i < num_of_vertices; i++) {
        if (i != id && outedge[i] != 0) {
            outc++;
        }
    }

    // first iteration
    if (iter > 0) {
        for (i = 0; i < num_of_vertices; i++) {
            // we don't consider self-loop
            if (inedge[i] && i != id) {
                sum += prev_pr[i];
            }
        }
        if (outc > 0) {
            curr_pr[id] = (RANDOMRESETPROB + (1 - RANDOMRESETPROB) * sum) / (double) outc;
        } else {
            curr_pr[id] = (RANDOMRESETPROB + (1 - RANDOMRESETPROB) * sum);
        }
    } else if (iter == 0) {
        if (outc > 0) {
            curr_pr[id] = 1.0f / (double) outc;
        }
    }

    // for the last iteration
    if (iter == niters - 1) {
        if (outc > 0) {
            vertices[id] = curr_pr[id] * (double) outc;
        } else {
            vertices[id] = curr_pr[id];
        }
    }
}

/**
 * TODO section:
 * maybe we can borrow the log system from graphchi?
 */
int main(int argc, char** argv) {
    int id, rc;
    size_t num_of_vertices, num_of_subvertices, niters;
    size_t stripe_size, iter, st, i;
    
    int64_t *graph, *outedges, *inedges;
    struct JSONRPCClient client;

    // result
    double *prev_pr, *curr_pr, *vertices;

    // timing
    struct timeval h_start, h_end;
    long duration;

    if (argc < 5) {
        printf("usage: %s <matrix id> <# of vertices> <# of subvertices> <niters>\n", argv[0]);
        exit(1);
    } 

    rc = connect_to_spdkrpc_server(&client);
    if (rc) {
        printf("cannot create conntection to SPDK RPC server");
        return rc;
    }

    graph = (int64_t *) mmap_to_tensorstore_hugepage();
    if (graph == NULL) {
        return -1;
    }

    id = atoi(argv[1]);
    num_of_vertices = (size_t) atoi(argv[2]);
    num_of_subvertices = (size_t) atoi(argv[3]);
    niters = (size_t) atoi(argv[4]);
    
    stripe_size = num_of_vertices * num_of_subvertices * sizeof(int64_t);
    // subgraph initialization
    outedges = (int64_t *) malloc(stripe_size);
    inedges = (int64_t *) malloc(stripe_size);

    // PR initialization
    vertices = (double *) calloc(sizeof(double), num_of_vertices);
    prev_pr = (double *) malloc(sizeof(double) * num_of_vertices);
    curr_pr = (double *) malloc(sizeof(double) * num_of_vertices);
    
    for (i = 0; i < num_of_vertices; i++) {
        curr_pr[i] = RANDOMRESETPROB;
    }

    memcpy(prev_pr, curr_pr, sizeof(double) * num_of_vertices);

    // Kernel loop (inspired by GraphChi engine)
    for (iter = 0; iter < niters; iter++) {
        printf("iter: %lu\n", iter);
        // userprogram.before_iteration(iter, chicontext);
        // do nothing in example

        // scheduler function
        // do nothing in example

        // shuffle function
        // do nothing in example

        // Interval loop
        // assume we have no subinterval in an interval.
        for (st = 0; st < (num_of_vertices / num_of_subvertices); st++) {
            /* preprocessing */

            // userprogram.before_exec_interval(interval_st, interval_en, chicontext);
            // do nothing in example

            // flush things back from sliding_shards

            // create a new memory shard
            tensorstore_get_row_stripe_submatrix(&client, id, st, st+1, num_of_subvertices);
            memcpy(outedges, graph, stripe_size);
            // get_outedges(graph, outedges, st, num_of_vertices_per_stripe, num_of_vertices);

            // inedges need to be transposed (?)
            // get_inedges(graph, inedges, st, st + num_of_vertices_per_stripe, num_of_vertices);

            // initialize vertices & edge data

            // load data
            // load_before_updates(vertices);

            /* execution part */

            // exec_updates(userprogram, vertices);
            // exec_updates(GraphChiProgram<VertexDataType, EdgeDataType, svertex_t> &userprogram, std::vector<svertex_t> &vertices)
            
            // update vertices one by one
            // userprogram.update(v, chicontext);
            // update(graphchi_vertex<VertexDataType, EdgeDataType> &v, graphchi_context &ginfo);
            for (i = 0; i < num_of_subvertices; i++) {
                pagerank_update(prev_pr, curr_pr, vertices, (st * num_of_subvertices) + i, (outedges + i * num_of_vertices), (outedges + i * num_of_vertices), num_of_vertices, iter, niters);
                // pagerank_update(prev_pr, curr_pr, vertices, i, (inedges + i * num_of_vertices), (outedges + i * num_of_vertices), num_of_vertices, iter, niters);
            }
            /* postprocessing */

            // save_vertices(vertices);
            // nothing is modified in PR application

            // memoryshard->commit(modifies_inedges, modifies_outedges & !disable_outedges);
            // memoryshard->commit(0, 0) in PR. doesn't need to commit back

            // userprogram.after_exec_interval(interval_st, interval_en, chicontext);
            // do nothing in PR
        }
        memcpy(prev_pr, curr_pr, sizeof(double) * num_of_vertices);
        // userprogram.after_iteration(iter, chicontext);
        // do nothing in PR

        // flush all stuff in sliding_shards
        // nothing inside maybe. it is just a buffer for updating
    }
    FILE *fp = fopen("log.txt", "w");
    for (i = 0; i < num_of_vertices; i++) {
        fprintf(fp, "%lu %f\n", i, vertices[i]);
    }
    fclose(fp);
    // cleanup
    munmap(graph, HUGEPAGE_SZ);
    free(outedges);
    free(inedges);
    free(vertices);
    free(prev_pr);
    free(curr_pr);

    return 0;
}