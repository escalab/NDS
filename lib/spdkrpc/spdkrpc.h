#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h> 
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/mman.h>

#define MAX_POLL_CQ_TIMEOUT 5000
#define BLOCK_SIZE 16384
#define HUGEPAGE_SZ 4 * 1024UL * 1024UL * 1024UL
#define SOCK_BUF_SZ 4096

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...)    fprintf(stderr, fmt, ## args)
#else
#define DEBUG_PRINT(fmt, args...)    /* Don't do anything in release builds */
#endif

struct JSONRPCClient {
    int verbose;
    int timeout;
    int request_id;
    int sock;
    char *addr;
    int port;
};

// sock.c
int spdk_rpc_connect(struct JSONRPCClient *client);

// json_parser.c
char *create_tensorstore_get_gather_matrix_json_string(struct JSONRPCClient* client, int id, int x, int y, int sub_m, char* rpc_method);
char *create_tensorstore_get_matrix_json_string(struct JSONRPCClient* client, int id, int x, int y);
size_t tensorstore_get_matrix_return_size(const char* respond_string);

// api.c
int connect_to_spdkrpc_server(struct JSONRPCClient *client);
void *mmap_to_tensorstore_hugepage(void);
size_t tensorstore_get_submatrix(struct JSONRPCClient *client, int id, int x, int y);
size_t tensorstore_get_gather_submatrix(struct JSONRPCClient *client, int id, int x, int y, int sub_m);
size_t tensorstore_get_row_stripe_submatrix(struct JSONRPCClient *client, int id, int y0, int y1, int sub_m);
size_t tensorstore_get_col_stripe_submatrix(struct JSONRPCClient *client, int id, int x0, int x1, int sub_n);
size_t tensorstore_write_row_stripe_submatrix(struct JSONRPCClient *client, int id, int y0, int y1, int sub_m);
size_t tensorstore_write_col_stripe_submatrix(struct JSONRPCClient *client, int id, int x0, int x1, int sub_n);