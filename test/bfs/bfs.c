#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

void kernel2(int* g_graph_mask, int *g_updating_graph_mask, int* g_graph_visited, int *g_over, int no_of_nodes) {
    uint64_t id;
    for (id = 0; id < no_of_nodes; id++) {
        if (g_updating_graph_mask[id]) {
            g_graph_mask[id] = 1;
            g_graph_visited[id] = 1;
            *g_over = 1;
            g_updating_graph_mask[id] = 0;
        }
    }
}

void kernel1(uint64_t *graph, int* g_graph_mask, int* g_updating_graph_mask, int *g_graph_visited, int* g_cost, int no_of_nodes) {
    uint64_t id, neighbor_id, offset;
    uint64_t *node;
    for (id = 0; id < no_of_nodes; id++) {
        if (g_graph_mask[id]) {
            g_graph_mask[id] = 0;
            offset = id * no_of_nodes;
            node = graph + offset;
            for (neighbor_id = 0; neighbor_id < no_of_nodes; neighbor_id++) {
                if (node[neighbor_id] && !g_graph_visited[neighbor_id]) {
                    g_cost[neighbor_id] = g_cost[id] + 1;
                    g_updating_graph_mask[neighbor_id] = 1;
                }
            }
        }
    }
}

void bfs(uint64_t *graph, int source, int num_nodes) {
    uint64_t i, j;
    int over = 1, count = 0;
    int *visited, *mask, *updating_mask, *cost;
    uint64_t *node;
    FILE *fpo;
    visited = calloc(num_nodes, sizeof(int));
    mask = calloc(num_nodes, sizeof(int));
    updating_mask = calloc(num_nodes, sizeof(int));
    cost = calloc(num_nodes, sizeof(int));

    for (i = 0; i < num_nodes; i++) {
        cost[i] = -1;
    }

    mask[source] = 1;
    visited[source] = 1;
    cost[source] = 0;

    printf("starting iteration\n");
    do {
        over = 0;
        kernel1(graph, mask, updating_mask, visited, cost, num_nodes);
        kernel2(mask, updating_mask, visited, &over, num_nodes);
        count++;
        printf("iteration %d\n", count);
    } while (over);

    //Store the result into a file
	fpo = fopen("result.txt","w");
	for (i = 0; i < num_nodes; i++) {
		fprintf(fpo, "%d) cost:%d\n", i, cost[i]);
    }
	fclose(fpo);
	printf("Result stored in result.txt\n");

    free(visited);
    free(mask);
    free(updating_mask);
    free(cost);
}

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

    bfs(graph, 0, num_nodes);
    close(fd);
    munmap(graph, size);
    return 0;
}