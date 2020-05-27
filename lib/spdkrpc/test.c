#include "spdkrpc.h"

#include <sys/stat.h>
#include <fcntl.h>

#define HUGEPAGE_SZ 4 * 1024UL * 1024UL * 1024UL
#define SOCK_BUF_SZ 4096

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

int main(int argc, char **argv) {
    // SPDK RPC part
    char *buf, *request_string, *original_data;
    int fd, rc;
    uint64_t i, j, k, checked_byte, offset, file_size, g_duration;
    struct stat st;
    struct timeval g_start, g_end;
    int matrix_size, submatrix_size, x, y;
    
    struct JSONRPCClient client;
    size_t return_size; 

    const char *hugepage_filename = "/dev/hugepages/tensorstore";
    int hugepage_fd;
    char *hugepage_addr;

    if (argc < 6) {
        printf("usage: %s <verify file path> <matrix size> <submatrix size> <x> <y>\n", argv[0]);
        exit(1);
    }

    matrix_size = atoi(argv[2]);
    submatrix_size = atoi(argv[3]);
    x = atoi(argv[4]);
    y = atoi(argv[5]);

    rc = connect_to_spdkrpc_server(&client);
    if (rc) {
        printf("cannot create conntection to SPDK RPC server");
        return rc;
    }

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

    memset(hugepage_addr, 0, HUGEPAGE_SZ);

    request_string = create_get_tensorstore_matrix_json_string(&client, 0, x, y);
    printf("%s\n", request_string);
    send(client.sock, request_string, strlen(request_string), 0);

    buf = calloc(SOCK_BUF_SZ, sizeof(char));

	gettimeofday(&g_start, NULL);
    recv(client.sock, buf, SOCK_BUF_SZ, 0);
    gettimeofday(&g_end, NULL);
    g_duration = ((g_end.tv_sec * 1000000 + g_end.tv_usec) - (g_start.tv_sec * 1000000 + g_start.tv_usec));
    printf("receive response elapsed time: %f s\n", (double) g_duration / 1000000);

    printf("response:\n %s\n", buf);

    gettimeofday(&g_start, NULL);
    return_size = get_tensorstore_matrix_return_size(buf);
    gettimeofday(&g_end, NULL);
    g_duration = ((g_end.tv_sec * 1000000 + g_end.tv_usec) - (g_start.tv_sec * 1000000 + g_start.tv_usec));
    printf("parse response JSON elapsed time: %f s\n", (double) g_duration / 1000000);

    // if (data == NULL) {
    //     printf("parse incorrect\n");
    //     exit(1);
    // }

    // assume matrix is a square
    fd = open(argv[1], O_RDONLY);
    fstat(fd, &st);
	file_size = st.st_size;
    original_data = (char *) mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    file_size = file_size / ((matrix_size / submatrix_size) * (matrix_size / submatrix_size));

    checked_byte = 0;
    offset = 0;
    if (file_size == return_size) {
        printf("compare data\n");
        for (i = y * submatrix_size; i < (y+1) * submatrix_size; i++) {
            for (j = x * submatrix_size; j < (x+1) * submatrix_size; j++) {
                // how many bytes per element needs to be checked
                for (k = 0; k < sizeof(double); k++) {
                    offset = (i*matrix_size+j)*sizeof(double)+k;
                    // printf("i=%lu, j=%lu, offset=%lu\n", i, j, offset);
                    if (hugepage_addr[checked_byte] != original_data[offset]) {
                        printf("data is wrong at byte %lu, data=%x, original_data=%x\n", checked_byte, hugepage_addr[checked_byte], original_data[offset]);
                        rc = -1;
                        abort();
                    }
                    checked_byte++;
                }
            }
        }
    } else {
        printf("return_size (%lu) != file_size (%lu)\n", return_size, file_size);
        rc = -1;
    }

    if (rc) {
        printf("Test Failed\n");
    } else {
        printf("Test Passed\n");
    }

    free(request_string);
    free(buf);
    // free(response);
    // free(data);

    close(fd);
    munmap(original_data, file_size);

    munmap(hugepage_addr, HUGEPAGE_SZ);
    close(hugepage_fd);
    return rc;
}