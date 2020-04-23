/******************************************************************************
  Socket operations
  For simplicity, the example program uses TCP sockets to exchange control
  information. If a TCP/IP stack/connection is not available, connection manager
  (CM) may be used to pass this information. Use of CM is beyond the scope of
  this example
 ******************************************************************************/

#include "spdkrpc.h"

/******************************************************************************
 * Function: sock_connect
 *
 * Input
 * servername URL of server to connect to (NULL for server mode)
 * port port of service
 *
 * Output
 * none
 *
 * Returns
 * socket (fd) on success, negative error code on failure
 *
 * Description
 * Connect a socket. If servername is specified a client connection will be
 * initiated to the indicated server and port. Otherwise listen on the
 * indicated port for an incoming connection.
 *
 ******************************************************************************/
int spdk_sock_connect(const char *servername, int port)
{
    int sockfd = -1;
    int tmp;
    unsigned long start_time_msec;
    unsigned long cur_time_msec;
    struct timeval cur_time;
    struct sockaddr_un addr;
    
    if ((sockfd = socket(AF_UNIX, SOCK_STREAM, 0)) == -1 ) {
        perror("socket error");
        return sockfd;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, servername, sizeof(addr.sun_path)-1);

    /* Client mode. Initiate connection to remote */
    gettimeofday(&cur_time, NULL);
    start_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
    do
    {
        tmp = connect(sockfd, (struct sockaddr*)&addr, sizeof(addr));
        gettimeofday(&cur_time, NULL);
        cur_time_msec = (cur_time.tv_sec * 1000) + (cur_time.tv_usec / 1000);
    } while ((tmp != 0) && ((cur_time_msec - start_time_msec) < MAX_POLL_CQ_TIMEOUT));
    if (tmp)        
    {
        fprintf(stderr, "failed connect \n");
        close(sockfd);
        sockfd = -1;
    }

    return sockfd;
}

int spdk_rpc_connect(struct JSONRPCClient *client)
{
    int rc = 0;
    DEBUG_PRINT("waiting on port %d for TCP connection\n",
                    client->port);

    client->sock = spdk_sock_connect(client->addr, client->port);

    if (client->sock < 0)
    {
        fprintf(stderr,
                "failed to establish TCP connection to server %s, port %d\n",
                client->addr, client->port);
        rc = -1;
    }
    
    DEBUG_PRINT("TCP connection was established\n");
    return rc;
}