//
// Created by Hank Liu on 2019-05-13.
//

#ifndef RDMA_H_
#define RDMA_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <getopt.h>
#include <inttypes.h>
#include <netdb.h>
#include <malloc.h>
#include <endian.h>
#include <byteswap.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <linux/falloc.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>

#define BUF_SIZE (4UL * 1024UL * 1024UL * 1024UL)
#define LENGTH (1024UL * 1024UL * 1024UL)
#define RDMA_BUF_SIZE 4096UL
#define SPDK_BUF_SIZE 512UL * 512UL * 8UL
#define CQ_NUM_ENTRIES 1024
#define MAX_POLL_CQ_TIMEOUT 5000
#define VERBOSE 0

#if __BYTE_ORDER==__LITTLE_ENDIAN
static inline uint64_t htonll(uint64_t x) {
return
bswap_64(x);
}
static inline uint64_t ntohll(uint64_t x) {
return
bswap_64(x);
}
#elif __BYTE_ORDER==__BIG_ENDIAN
static inline uint64_t htonll(uint64_t x) { return x; }
static inline uint64_t ntohll(uint64_t x) { return x; }
#else
#error __BYTE_ORDER is neither __LITTLE_ENDIAN nor __BIG_ENDIAN
#endif

/* structure of test parameters */
struct config_t {
  const char *dev_name; /* IB device name */
  char *server_name;    /* server host name */
  u_int32_t tcp_port;   /* server TCP port */
  int ib_port;          /* local IB port to work with */
  int gid_idx;          /* gid index to use */
};

/* structure to exchange data which is needed to connect the QPs */
struct cm_con_data_t {
  uint64_t addr;   /* Buffer address */
  uint32_t rkey;   /* Remote key */
  uint32_t qp_num; /* QP number */
  uint16_t lid;    /* LID of the IB port */
  uint8_t gid[16]; /* gid */
} __attribute__((packed));

/* structure of system resources */
struct resources {
  struct ibv_device_attr device_attr;
  /* Device attributes */
  struct ibv_port_attr port_attr;    /* IB port attributes */
  struct cm_con_data_t remote_props; /* values to connect to remote side */
  struct ibv_context *ib_ctx;        /* device handle */
  struct ibv_pd *pd;                 /* PD handle */
  struct ibv_cq *cq;                 /* CQ handle */
  struct ibv_qp *qp;                 /* QP handle */
  struct ibv_mr *mr;                 /* MR handle for buf */
  char *buf;                         /* memory buffer pointer, used for RDMA and send
                                     ops */
  uint64_t size;
  int sock;                          /* TCP socket file descriptor */
};

void config_destroy(struct config_t *config);
int resources_destroy(struct resources *res);
void resources_init(struct resources *res);
int resources_find_device(struct resources *res, struct config_t *config);
int resources_create(struct resources *res, struct config_t *config);
int make_tcp_connection(struct resources *res, struct config_t *config);
int connect_qp(struct resources *res, struct config_t *config);
int sock_sync_data(int sock, int xfer_size, char *local_data, char *remote_data);
int sock_write_data(int sock);
int sock_read_data(int sock);
int poll_completion(struct resources *res, uint64_t num_entries);
int post_send(struct resources *res, int opcode, char *spdk_ptr, uint64_t remote_offset, uint32_t length);
int post_send_multi_sr(struct resources *res, int opcode, char *spdk_ptr, uint64_t remote_offset, uint64_t rows, uint64_t remote_row_size, uint32_t length);
int post_send_nds(struct resources *res, int opcode, char *spdk_ptr_0, char *spdk_ptr_1, uint64_t remote_offset, uint64_t rows, uint64_t remote_row_size, uint32_t length);
int post_receive(struct resources *res);
#endif //RDMA_H_
