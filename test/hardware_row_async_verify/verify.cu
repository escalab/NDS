extern "C" {
    #include "rdma.h"
    #include "timing.h"
    #include "fifo.h"
}

#define RANDOMRESETPROB 0.15
#define NUM_THREADS 1024

#define HUGEPAGE_SZ (4UL * 1024UL * 1024UL * 1024UL)
#define M 65536UL
#define SUB_M 4096UL
#define AGGREGATED_SZ (M * SUB_M * 8UL)

#define IO_QUEUE_SZ (HUGEPAGE_SZ / AGGREGATED_SZ)
// #define IO_QUEUE_SZ 1UL

#define NITERS 4UL

void print_config(struct config_t config);

struct fetch_conf {
    struct resources *res;
    uint64_t m, sub_m;
    int64_t *row_stripe;
    char *hugepage_addr;
    struct fifo *sending_queue;
    struct fifo *complete_queue;
};

struct fifo_entry {
    int64_t *row_stripe;
};

int cudaMemcpyFromMmap(struct fetch_conf *conf, char *dst, const char *src, const size_t length) {
    struct response *res = NULL;

    if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
        fprintf(stderr, "sync error before RDMA ops\n");
        return 1;
    }
    res = sock_read_offset(conf->res->sock);
    if (res == NULL) {
        fprintf(stderr, "sync error before RDMA ops\n");
        return 1;
    }

    // if (res->id == 0) {
    //     printf("fetching row [%lu:%lu]\n", res->x, res->y);
    // } else {
    //     printf("fetching col [%lu:%lu]\n", res->x, res->y);
    // }
    // printf("offset: %lu, length: %lu\n", res->offset, length);

    memcpy(dst, src + res->offset, length);

    free(res);
    if (sock_write_data(conf->res->sock)) { /* just send a dummy char back and forth */
        fprintf(stderr, "sync error before RDMA ops\n");
        return 1;
    }
    if (sock_read_data(conf->res->sock)) { /* just send a dummy char back and forth */
        fprintf(stderr, "sync error before RDMA ops\n");
        return 1;
    }
    return 0;
}

void *fetch_thread(void *args) {
    struct fetch_conf *conf = (struct fetch_conf *) args;
    uint64_t i, st;
    uint64_t dsize = conf->m * conf->sub_m;
    int64_t *ptr_a;
    struct fifo_entry *entry = NULL;
    uint64_t count = 0;

    for (i = 0; i < NITERS; i++) {
        for (st = 0; st < conf->m / conf->sub_m; st++) {
            while (fifo_empty(conf->complete_queue)) {

            }
            entry = (struct fifo_entry *) fifo_pop(conf->complete_queue);
            ptr_a = conf->row_stripe + dsize * (count % IO_QUEUE_SZ);

            cudaMemcpyFromMmap(conf, (char *) ptr_a, (char *) conf->hugepage_addr, dsize * sizeof(int64_t));
            count++;

            entry->row_stripe = ptr_a;
            fifo_push(conf->sending_queue, entry);
        }
    }
    return NULL;
}

uint64_t row_verify(int64_t *arr, int64_t *answer, uint64_t st, uint64_t m, uint64_t sub_m) {
    uint64_t i, j, error = 0;
    for (i = 0; i < sub_m; i++) {
        // printf("arr[%lu], answer[%lu]\n", i, i+st);
        for (j = 0; j < m; j++) {
            if (arr[i * m + j] != answer[(i+st) * m + j]) {
                error++;
            }
        }
    }
    return error;
}

int nds_pagerank(struct resources *res, uint64_t m, uint64_t sub_m, int64_t *answer) {
    size_t i, st;    
    int64_t *row_stripe;
    
    uint64_t stripe_size;
    uint64_t error = 0;

    struct fifo *sending_queue;
    struct fifo *complete_queue; 
    struct fifo_entry *entries = (struct fifo_entry *) calloc(IO_QUEUE_SZ, sizeof(struct fifo_entry));
    struct fifo_entry *entry = NULL;

    pthread_t thread_id; 
    struct fetch_conf conf;

    // it causes problem if size == 1
    sending_queue = fifo_new(IO_QUEUE_SZ * 2);
	if (sending_queue == NULL) {
        printf("cannot create sending_queue\n");
        return -1;
    }
    
    complete_queue = fifo_new(IO_QUEUE_SZ * 2);
	if (complete_queue == NULL) {
        printf("cannot create complete_queue\n");
        return -1;
    }
    
    for (i = 0; i < IO_QUEUE_SZ; i++) {
        fifo_push(complete_queue, entries + i);
    }

    // subgraph initialization
    stripe_size = m * sub_m * sizeof(int64_t);
    row_stripe = (int64_t *) malloc(stripe_size * IO_QUEUE_SZ);

    // create thread here
    conf.res = res;
    conf.m = m;
    conf.sub_m = sub_m;
    conf.row_stripe = row_stripe;
    conf.hugepage_addr = res->buf;
    conf.sending_queue = sending_queue;
    conf.complete_queue = complete_queue;

	pthread_create(&thread_id, NULL, fetch_thread, &conf); 
    // blockGEMM
    for (i = 0; i < NITERS; i++) {
        printf("iter: %lu\n", i);
        for (st = 0; st < m / sub_m; st++) {
            printf("st: %lu\n", st * sub_m);
            while (fifo_empty(sending_queue)) {

            }
            entry = (struct fifo_entry *) fifo_pop(sending_queue);
            error += row_verify(entry->row_stripe, answer, st * sub_m, m, sub_m);
            fifo_push(complete_queue, entry);
        }
        printf("row_error: %lu\n", error);
    }

    pthread_join(thread_id, NULL); 

    free(row_stripe);
    fifo_free(sending_queue);
    fifo_free(complete_queue);
    free(entries);
    return 0;
}

/******************************************************************************
 * Function: print_config
 *
 * Input
 * none
 *
 * Output
 * none
 *
 * Returns
 * none
 *
 * Description
 * Print out config information
 ******************************************************************************/
void print_config(struct config_t config) {
    fprintf(stdout, " ------------------------------------------------\n");
    fprintf(stdout, " Device name : \"%s\"\n", config.dev_name);
    fprintf(stdout, " IB port : %u\n", config.ib_port);
    if (config.server_name)
        fprintf(stdout, " IP : %s\n", config.server_name);
    fprintf(stdout, " TCP port : %u\n", config.tcp_port);
    if (config.gid_idx >= 0)
        fprintf(stdout, " GID index : %u\n", config.gid_idx);
    fprintf(stdout, " ------------------------------------------------\n\n");
}

int main(int argc, char *argv[]) {
    int rc = 0;
    int hugepage_fd;
    char *hugepage_addr;

    int answer_fd;
    char *answer_addr;

    // RDMA
    struct resources res;
    struct config_t config = {
        "mlx4_0",  /* dev_name */
        NULL,  /* server_name */
        19875, /* tcp_port */
        1,     /* ib_port */
        0     /* gid_idx */
    };

    if (argc < 3) {
        printf("usage: %s <answer path> <port>\n", argv[0]);
        exit(1);
    } 

    config.tcp_port = atoi(argv[2]);

    /* print the used parameters for info*/
    print_config(config);
    
    printf("mapping hugepage\n");
    hugepage_fd = open("/dev/hugepages/tensorstore", O_RDWR, 0755);
    if (hugepage_fd < 0) {
        perror("open");
        exit(1);
    }

    hugepage_addr = (char *) mmap(0, BUF_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, hugepage_fd, 0);
    if (hugepage_addr==MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    answer_fd = open(argv[1], O_RDONLY, 0755);
    if (answer_fd < 0) {
        perror("open");
        exit(1);
    }

    answer_addr = (char *) mmap(0, M*M*sizeof(int64_t), PROT_READ, MAP_PRIVATE, answer_fd, 0);
    if (answer_addr==MAP_FAILED) {
        perror("mmap");
        exit(1);
    }

    res.buf = hugepage_addr;
    memset(hugepage_addr, 0, BUF_SIZE);

    printf("hugepage starting address is: %p\n", hugepage_addr);
    printf("socket connection\n");
    rc = make_tcp_connection(&res, &config);
    if (rc < 0) {
        perror("sock connect");
        exit(1);
    }

    // use the first res to find the device.
    if (resources_find_device(&res, &config)) {
        fprintf(stderr, "failed to find a device for resources\n");
        exit(1);
    }

    if (resources_create(&res, &config)) {
        fprintf(stderr, "failed to create resources\n");
        exit(1);
    }

    /* connect the QPs */
    if (connect_qp(&res, &config)) {
        fprintf(stderr, "failed to connect QPs\n");
        exit(1);
    }

    printf("calculating the result of pagerank\n");
    rc = nds_pagerank(&res, M, SUB_M, (int64_t *) answer_addr);
    
    // close(res.sock);
    if (resources_destroy(&res)) {
        fprintf(stderr, "failed to destroy resources\n");
        exit(1);
    }
    munmap(hugepage_addr, BUF_SIZE);
    close(hugepage_fd);
    munmap(answer_addr, M*M*sizeof(int64_t));
    close(answer_fd);
    return rc;
}
