#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// for mmap
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

void bfs(uint64_t *graph, int source, int num_nodes) {
    uint64_t i, j;
    int next_round = 1, count = 0;
    int *visited, *curr, *next;
    uint64_t *node;
    visited = calloc(num_nodes, sizeof(int));
    curr = calloc(num_nodes, sizeof(int));
    next = calloc(num_nodes, sizeof(int));
    curr[source] = 1;
    printf("starting iteration\n");
    while (next_round) {
        next_round = 0;
        // iterate current nodes in the array
        for (i = 0; i < num_nodes; i++) {
            // if the node hasn't been visited yet.
            if (curr[i] && !visited[i]) {
                node = graph + (i * num_nodes);

                // add the neighbors of the current node to the next array.
                for (j = 0; j < num_nodes; j++) {
                    if (node[j]) {
                        next[j] = 1;
                        // indicate we will have next iteration.
                        next_round = 1;
                    }
                }
                // finished visit, mark the node.
                visited[i] = 1;
            }
        }
        memcpy(curr, next, num_nodes * sizeof(int));
        memset(next, 0, num_nodes * sizeof(int));
        count++;
        printf("iteration %d\n", count);
    }
    count = 0;
    for (i = 0; i < num_nodes; i++) {
        if (visited[i]) {
            count++;
        }
    }

    printf("num of visited nodes: %d\n", count);
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