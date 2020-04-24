#include "spdkrpc.h"

#include <sys/stat.h>
#include <fcntl.h>

#define HUGEPAGE_SZ 4 * 1024UL * 1024UL * 1024UL
#define SOCK_BUF_SZ 4096

int main(int argc, char **argv) {
    // SPDK RPC part
    char *buf, *request_string, *original_data;
    int fd, rc;
    uint64_t i, file_size, g_duration;
    struct stat st;
    struct timeval g_start, g_end;

    struct JSONRPCClient client = {
        .verbose = 0,
        .timeout = 60,
        .request_id = 0,
        .addr = "/var/tmp/spdk.sock",
        .port = 5260
    };

    size_t return_size; 

    char *hugepage_filename = "/dev/hugepages/tensorstore";
    int hugepage_fd;
    char *hugepage_addr;

    if (argc < 2) {
        printf("usage: %s <verify file path>\n", argv[0]);
        exit(1);
    }

    rc = spdk_rpc_connect(&client);
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

    request_string = create_get_tensorstore_matrix_json_string(&client, 1, 0, 0);
    printf("%s\n", request_string);
    send(client.sock, request_string, strlen(request_string), 0);

    buf = calloc(SOCK_BUF_SZ, sizeof(char));

	gettimeofday(&g_start, NULL);
    recv(client.sock, buf, SOCK_BUF_SZ, 0);
    gettimeofday(&g_end, NULL);
    g_duration = ((g_end.tv_sec * 1000000 + g_end.tv_usec) - (g_start.tv_sec * 1000000 + g_start.tv_usec));
    printf("receive response elapsed time: %f s\n", (double) g_duration / 1000000);

    // printf("response:\n %s\n", buf);

    gettimeofday(&g_start, NULL);
    return_size = get_tensorstore_matrix_return_size(buf);
    gettimeofday(&g_end, NULL);
    g_duration = ((g_end.tv_sec * 1000000 + g_end.tv_usec) - (g_start.tv_sec * 1000000 + g_start.tv_usec));
    printf("parse response JSON elapsed time: %f s\n", (double) g_duration / 1000000);

    // if (data == NULL) {
    //     printf("parse incorrect\n");
    //     exit(1);
    // }

    fd = open(argv[1], O_RDONLY);
    fstat(fd, &st);
	file_size = st.st_size;
    original_data = (char *) mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    if (file_size == return_size) {
        printf("compare data\n");
        for (i = 0; i < return_size; i++) {
            if (hugepage_addr[i] != original_data[i]) {
                printf("data is wrong at byte %lu\n", i);
                rc = -1;
                break;
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