/*
 * daemonize.c
 * This example daemonizes a process, writes a few log messages,
 * sleeps 20 seconds and terminates afterwards.
 * This is an answer to the stackoverflow question:
 * https://stackoverflow.com/questions/17954432/creating-a-daemon-in-linux/17955149#17955149
 * Fork this code: https://github.com/pasce/daemon-skeleton-linux-c
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <syslog.h>
#include <rdma.h>

int main(int argc, char** argv)
{
    struct resources res;
    struct config_t config = {
        "mlx4_0",  /* dev_name */
        "127.0.0.1",  /* server_name */
        19875, /* tcp_port */
        1,     /* ib_port */
        0     /* gid_idx */
    };
    int rc;
    struct request *req_from_host;

    if (argc < 2) {
        printf("usage: %s [software=0|hardware=1|terminate=2]\n", argv[0]);
        exit(1);
    }

    rc = make_two_tcp_connection(&res, &config);
    if (rc < 0) {
        perror("sock connect");
        exit(1);
    }

    sock_write_request(res.req_sock, atoi(argv[1]), 0, 0, 4096, 1, 0);
    sock_read_data(res.req_sock);

    close(res.sock);
    close(res.req_sock); 

    return EXIT_SUCCESS;
}
