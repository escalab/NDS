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

// mmap_reader.c
char* read_from_spdk(int pid, unsigned long file_size, int parts, unsigned long *addr_list, char *buf); 

// sock.c
int spdk_rpc_connect(struct JSONRPCClient *client);

// json_parser.c
char *create_get_tensorstore_matrix_json_string(struct JSONRPCClient* client, int id, int x, int y);
char *parse_get_tensorstore_matrix_json(const char* respond_string, int pid);