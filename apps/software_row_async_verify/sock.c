//
// Created by Hank Liu on 2019-05-13.
//

#include "rdma.h"

int sock_connect(const char *servername, int port);

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
int sock_connect(const char *servername, int port) {
    struct addrinfo *resolved_addr = NULL;
    struct addrinfo *iterator;
    char service[6];
    int sockfd = -1;
    int listenfd = 0;
    int tmp;
    unsigned long start_time_msec;
    unsigned long cur_time_msec;
    struct timeval cur_time;
    struct addrinfo hints = {
        .ai_flags = AI_PASSIVE,
        .ai_family = AF_INET,
        .ai_socktype = SOCK_STREAM};
    if (sprintf(service, "%d", port) < 0)
        goto sock_connect_exit;
    /* Resolve DNS address, use sockfd as temp storage */
    sockfd = getaddrinfo(servername, service, &hints, &resolved_addr);
    if (sockfd < 0) {
        fprintf(stderr, "%s for %s:%d\n", gai_strerror(sockfd), servername,
                port);
        goto sock_connect_exit;
    }
    /* Search through results and find the one we want */
    for (iterator = resolved_addr; iterator; iterator = iterator->ai_next) {
        sockfd =
            socket(iterator->ai_family, iterator->ai_socktype,
                   iterator->ai_protocol);
        if (sockfd >= 0) {
            if (servername) {
                /* Client mode. Initiate connection to remote */
                gettimeofday(&cur_time, NULL);
                start_time_msec = (cur_time.tv_sec*1000) + (cur_time.tv_usec/1000);
                do {
                    tmp = connect(sockfd, iterator->ai_addr, iterator->ai_addrlen);
                    gettimeofday(&cur_time, NULL);
                    cur_time_msec = (cur_time.tv_sec*1000) + (cur_time.tv_usec/1000);
                } while ((tmp!=0) && ((cur_time_msec - start_time_msec) < MAX_POLL_CQ_TIMEOUT));
                if (tmp) {
                    fprintf(stdout, "failed connect \n");
                    close(sockfd);
                    sockfd = -1;
                }
            } else {
                /* Server mode. Set up listening socket an accept a connection */
                listenfd = sockfd;
                sockfd = -1;
                if (bind(listenfd, iterator->ai_addr, iterator->ai_addrlen))
                    goto sock_connect_exit;
                listen(listenfd, 1);
                sockfd = accept(listenfd, NULL, 0);
            }
        }
    }
    sock_connect_exit:
    if (listenfd)
        close(listenfd);
    if (resolved_addr)
        freeaddrinfo(resolved_addr);
    if (sockfd < 0) {
        if (servername)
            fprintf(stderr, "Couldn't connect to %s:%d\n", servername, port);
        else {
            perror("server accept");
            fprintf(stderr, "accept() failed\n");
        }
    }
    return sockfd;
}

int make_tcp_connection(struct resources *res, struct config_t *config) {
    int rc = 0;
    /* if client side */
    if (config->server_name) {
        res->sock = sock_connect(config->server_name, config->tcp_port);
        if (res->sock < 0) {
            fprintf(stderr,
                    "failed to establish TCP connection to server %s, port %d\n",
                    config->server_name, config->tcp_port);
            rc = -1;
        }
    }
        // if server side
    else {
#if VERBOSE==1
        fprintf(stdout, "waiting on port %d for TCP connection\n",
                config->tcp_port);
#endif
        res->sock = sock_connect(NULL, config->tcp_port);
        if (res->sock < 0) {
            fprintf(stderr,
                    "failed to establish TCP connection with client on port %d\n",
                    config->tcp_port);
            rc = -1;
        }
    }
#if VERBOSE==1
    fprintf(stdout, "TCP connection was established\n");
#endif
    return rc;
}

/******************************************************************************
 * Function: sock_sync_data
 *
 * Input
 * sock socket to transfer data on
 * xfer_size size of data to transfer
 * local_data pointer to data to be sent to remote
 *
 * Output
 * remote_data pointer to buffer to receive remote data
 *
 * Returns
 * 0 on success, negative error code on failure
 *
 * Description
 * Sync data across a socket. The indicated local data will be sent to the
 * remote. It will then wait for the remote to send its data back. It is
 * assumed that the two sides are in sync and call this function in the proper
 * order. Chaos will ensue if they are not. :)
 *
 * Also note this is a blocking function and will wait for the full data to be
 * received from the remote.
 *
 ******************************************************************************/
int sock_sync_data(int sock, int xfer_size, char *local_data, char *remote_data) {
    int rc;
    int read_bytes = 0;
    int total_read_bytes = 0;
    rc = write(sock, local_data, xfer_size);
    if (rc < xfer_size)
        fprintf(stderr, "Failed writing data during sock_sync_data\n");
    else
        rc = 0;
    while (!rc && total_read_bytes < xfer_size) {
        read_bytes = read(sock, remote_data, xfer_size);
        remote_data[read_bytes] = '\0';
        if (read_bytes > 0)
            total_read_bytes += read_bytes;
        else
            rc = read_bytes;
    }
    return rc;
}

int sock_write_data(int sock) {
    int rc = 0;
    char dummy = 's';

    rc = write(sock, &dummy, sizeof(char));
    if (rc < (int) sizeof(char))
        fprintf(stderr, "Failed writing data during sock_sync_data\n");
    else
        rc = 0;
    return rc;
}

int sock_read_data(int sock) {
    int rc = 0;
    int read_bytes = 0;
    int total_read_bytes = 0;
    char dummy;
    while (!rc && total_read_bytes < (int) sizeof(char)) {
        read_bytes = read(sock, &dummy, sizeof(char));
        if (read_bytes > 0)
            total_read_bytes += read_bytes;
        else
            rc = read_bytes;
    }
    return rc;
}

int sock_write_response(int sock, uint64_t id, uint64_t x, uint64_t y, uint64_t offset) {
    int rc = 0;
    struct response res;
    res.id = id;
    res.x = x;
    res.y = y;
    res.offset = offset;

    rc = write(sock, &res, sizeof(struct response));
    if (rc < (int) sizeof(struct response))
        fprintf(stderr, "Failed writing data during sock_sync_data\n");
    else
        rc = 0;
    return rc;
}

struct response *sock_read_offset(int sock) {
    int rc = 0;
    int read_bytes = 0;
    int total_read_bytes = 0;
    struct response *res = calloc(1, sizeof(struct response));
    while (!rc && total_read_bytes < (int) sizeof(struct response)) {
        read_bytes = read(sock, res, sizeof(struct response));
        if (read_bytes > 0) {
            total_read_bytes += read_bytes;
        }
        else {
            free(res);
            res = NULL;
        }
    }
    return res;
}
