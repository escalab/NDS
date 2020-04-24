#include "spdkrpc.h"

#include <sys/stat.h>
#include <fcntl.h>

int main(int argc, char **argv) {
    // SPDK RPC part
    char *request_string, *data, *original_data;
    char *buf, *response, *ptr;
    int pid, fd, rc;
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
    int numBytesRecv = 0;
    size_t buf_size = 4096; 

    if (argc < 3) {
        printf("usage: %s <SPDK Server PID> <verify file path>\n", argv[0]);
        exit(1);
    }

    pid = atoi(argv[1]);

    rc = spdk_rpc_connect(&client);
    if (rc) {
        printf("cannot create conntection to SPDK RPC server");
        return rc;
    }

    request_string = create_get_tensorstore_matrix_json_string(&client, 0, 0, 0);
    printf("%s\n", request_string);
    send(client.sock, request_string, strlen(request_string), 0);

    buf = calloc(buf_size, sizeof(char));
    response = calloc(33554432, sizeof(char));    
    ptr = response;

	gettimeofday(&g_start, NULL);
    // TODO: how to know the last receive is coming?
    do {
        memset(buf, 0, buf_size);
        numBytesRecv = recv(client.sock, buf, buf_size, 0);
        // printf("numBytesRecv %d\n", numBytesRecv);
        if (numBytesRecv < 0) {
            printf("error receive\n");
            exit(1);
        }
        memcpy(ptr, buf, numBytesRecv);
        ptr += numBytesRecv;
    } while (numBytesRecv == buf_size || numBytesRecv == 2176);

    gettimeofday(&g_end, NULL);
    g_duration = ((g_end.tv_sec * 1000000 + g_end.tv_usec) - (g_start.tv_sec * 1000000 + g_start.tv_usec));
    printf("receive response elapsed time: %f s\n", (double) g_duration / 1000000);

    // {
    //     // printf("numBytesRecv %d\n", numBytesRecv);
    //     memset(buf, 0, buf_size);
    //     numBytesRecv = recv(client.sock, buf, buf_size, 0);
    //     if (numBytesRecv < 0) {
    //         printf("error receive\n");
    //         exit(1);
    //     }
    //     if (numBytesRecv == 0) {
    //         break;
    //     }
    //     memcpy(ptr, buf, numBytesRecv);
    //     ptr += numBytesRecv;
    // }

    // printf("response %s\n", response);
    gettimeofday(&g_start, NULL);
    data = parse_get_tensorstore_matrix_json(response, pid);
    gettimeofday(&g_end, NULL);
    g_duration = ((g_end.tv_sec * 1000000 + g_end.tv_usec) - (g_start.tv_sec * 1000000 + g_start.tv_usec));
    printf("parse response and read the memory map elapsed time: %f s\n", (double) g_duration / 1000000);

    if (data == NULL) {
        printf("parse incorrect\n");
        exit(1);
    }

    fd = open(argv[2], O_RDONLY);
    fstat(fd, &st);
	file_size = st.st_size;
    original_data = (char *) mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    printf("compare data\n");
    for (i = 0; i < file_size; i++) {
        if (data[i] != original_data[i]) {
            printf("data is wrong at byte %lu\n", i);
            rc = -1;
            break;
        }
    }

    if (rc) {
        printf("Test Failed\n");
    } else {
        printf("Test Passed\n");
    }

    free(request_string);
    free(buf);
    free(response);
    free(data);

    close(fd);
    munmap(original_data, file_size);
    return rc;
}