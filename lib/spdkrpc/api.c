#include "spdkrpc.h"

void construct_spdkrpc_client(struct JSONRPCClient *client) {
    client->verbose = 0;
    client->timeout = 60;
    client->request_id = 0;
    client->addr = "/var/tmp/spdk.sock";
    client->port = 5260;
}

int connect_to_spdkrpc_server(struct JSONRPCClient *client) {
    construct_spdkrpc_client(client);
    return spdk_rpc_connect(client);
}

void *mmap_to_tensorstore_hugepage(void) {
    const char *hugepage_filename = "/dev/hugepages/tensorstore";
    int hugepage_fd;
    void *hugepage_addr;
    
    hugepage_fd = open(hugepage_filename, O_RDWR, 0755);
    if (hugepage_fd < 0) {
        perror("open");
        exit(1);
    }

    hugepage_addr = mmap(0, HUGEPAGE_SZ, PROT_READ | PROT_WRITE, MAP_SHARED, hugepage_fd, 0);
    if (hugepage_addr == MAP_FAILED) {
        perror("mmap");
        unlink(hugepage_filename);
        exit(1);
    }
    close(hugepage_fd);

    memset(hugepage_addr, 0, HUGEPAGE_SZ);
    return hugepage_addr;
}

size_t send_request_and_get_returned_size(struct JSONRPCClient *client, char *request_string) {
    char *buf;
    size_t return_size; 
    // struct timeval g_start, g_end;
    // uint64_t g_duration;

    // printf("%s\n", request_string);
    send(client->sock, request_string, strlen(request_string), 0);

    buf = calloc(SOCK_BUF_SZ, sizeof(char));

	// gettimeofday(&g_start, NULL);
    recv(client->sock, buf, SOCK_BUF_SZ, 0);
    // gettimeofday(&g_end, NULL);
    // g_duration = ((g_end.tv_sec * 1000000 + g_end.tv_usec) - (g_start.tv_sec * 1000000 + g_start.tv_usec));
    // printf("receive response elapsed time: %f s\n", (double) g_duration / 1000000);

    // printf("response:\n %s\n", buf);

    // gettimeofday(&g_start, NULL);
    return_size = tensorstore_get_matrix_return_size(buf);
    // gettimeofday(&g_end, NULL);
    // g_duration = ((g_end.tv_sec * 1000000 + g_end.tv_usec) - (g_start.tv_sec * 1000000 + g_start.tv_usec));
    // printf("parse response JSON elapsed time: %f s\n", (double) g_duration / 1000000);
    free(buf);

    return return_size;
}

size_t tensorstore_get_submatrix(struct JSONRPCClient *client, int id, int x, int y) {
    char *request_string;
    size_t return_size; 

    request_string = create_tensorstore_get_matrix_json_string(client, id, x, y);
    return_size = send_request_and_get_returned_size(client, request_string);
    
    free(request_string);

    return return_size;
}

size_t tensorstore_get_gather_submatrix(struct JSONRPCClient *client, int id, int x, int y, int sub_m) {
    char *request_string;
    size_t return_size; 

    request_string = create_tensorstore_get_gather_matrix_json_string(client, id, x, y, sub_m);
    return_size = send_request_and_get_returned_size(client, request_string);
    
    free(request_string);

    return return_size;
}