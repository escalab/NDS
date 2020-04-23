#include "spdkrpc.h"

int main(int argc, char **argv) {
    // SPDK RPC part
    char *request_string, *data, *original_data;
    char *buf;
    int pid;

    struct JSONRPCClient client = {
        .verbose = 0,
        .timeout = 60,
        .request_id = 0,
        .addr = "/var/tmp/spdk.sock",
        .port = 5260
    };

    if (argc < 2) {
        printf("usage: %s <SPDK Server PID>\n", argv[0]);
        exit(1);
    }

    pid = atoi(argv[1]);

    if (spdk_rpc_connect(&client)) {
        printf("cannot create conntection to SPDK RPC server");
        return -1;
    }

    request_string = create_get_tensorstore_matrix_json_string(&client, 0, 0, 0);
    printf("%s\n", request_string);
    send(client.sock, request_string, strlen(request_string), 0);

    buf = calloc(524288, sizeof(char));
    recv(client.sock, buf, 262144, 0);

    data = parse_get_tensorstore_matrix_json(buf, pid);
    
    printf("%f\n", ((double *)data)[0]);

    free(request_string);
    free(buf);
    free(data);
    return 0;
}