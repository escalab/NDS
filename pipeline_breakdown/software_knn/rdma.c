//
// Created by Hank Liu on 2019-05-16.
//

#include "rdma.h"

/* poll_completion */
/******************************************************************************
 * Function: poll_completion
 *
 * Input
 * res pointer to resources structure
 *
 * Output
 * none
 *
 * Returns
 * 0 on success, 1 on failure
 *
 * Description
 * Poll the completion queue for a single event. This function will continue to
 * poll the queue until MAX_POLL_CQ_TIMEOUT milliseconds have passed.
 *
 ******************************************************************************/
int poll_completion(struct resources *res, uint64_t num_entries) {
    struct ibv_wc wc;
    unsigned long start_time_msec;
    unsigned long cur_time_msec;
    struct timeval cur_time;
    int poll_result = 0;
    int rc = 0;
    
    /* poll the completion for a while before giving up of doing it .. */
    gettimeofday(&cur_time, NULL);
    start_time_msec = (cur_time.tv_sec*1000) + (cur_time.tv_usec/1000);
    do {
        poll_result += ibv_poll_cq(res->cq, 1, &wc);
        gettimeofday(&cur_time, NULL);
        cur_time_msec = (cur_time.tv_sec*1000) + (cur_time.tv_usec/1000);
    } while ((poll_result < num_entries) && ((cur_time_msec - start_time_msec) < MAX_POLL_CQ_TIMEOUT));
    
    if (poll_result < 0) {
        /* poll CQ failed */
        fprintf(stderr, "poll CQ failed\n");
        rc = 1;
    } else if (poll_result==0) {
        /* the CQ is empty */
        fprintf(stderr, "completion wasn't found in the CQ after timeout\n");
        rc = 1;
    } else {
#if VERBOSE==1
        /* CQE found */
        fprintf(stdout, "completion was found in CQ with status 0x%x\n",
                wc.status);
#endif
        /* check the completion status (here we don't care about the completion opcode */
        if (wc.status!=IBV_WC_SUCCESS) {
            fprintf(stderr,
                    "got bad completion with status: 0x%x, vendor syndrome: 0x%x\n",
                    wc.status, wc.vendor_err);
            rc = 1;
        }
        // fprintf(stdout, "poll result = %d\n", poll_result);
    }
    return rc;
}

/******************************************************************************
 * Function: post_send_nds
 *
 * Input
 * res pointer to resources structure
 * opcode IBV_WR_SEND, IBV_WR_RDMA_READ or IBV_WR_RDMA_WRITE
 *
 * Output
 * none
 *
 * Returns
 * 0 on success, error code on failure
 *
 * Description
 * This function will create and post a send work request
 ******************************************************************************/
int post_send_nds(struct resources *res, int opcode, char *spdk_ptr_0, char *spdk_ptr_1, uint64_t remote_offset, uint64_t rows, uint64_t remote_row_size, uint32_t length) {
    struct ibv_send_wr *sr;
    struct ibv_sge *sge;
    struct ibv_send_wr *bad_wr = NULL;
    int rc;
    uint64_t i;

    sr = calloc(rows, sizeof(struct ibv_send_wr));
    sge = calloc(rows * 2, sizeof(struct ibv_sge));
    for (i = 0; i < rows - 1; i++) {
        /* prepare the scatter/gather entry */
        sge[i * 2].addr = (uintptr_t) spdk_ptr_0 + i * length;
        sge[i * 2].length = length;
        sge[i * 2].lkey = res->mr->lkey;
        sge[i * 2 + 1].addr = (uintptr_t) spdk_ptr_1 + i * length;
        sge[i * 2 + 1].length = length;
        sge[i * 2 + 1].lkey = res->mr->lkey;

        /* prepare the send work request */
        sr[i].next = &sr[i+1];
        sr[i].wr_id = i;
        sr[i].sg_list = &sge[2 * i];
        sr[i].num_sge = 2;
        sr[i].opcode = opcode;
        sr[i].send_flags = IBV_SEND_SIGNALED;
        if (opcode!=IBV_WR_SEND) {
            sr[i].wr.rdma.remote_addr = res->remote_props.addr + remote_offset;
            sr[i].wr.rdma.rkey = res->remote_props.rkey;
        }
        remote_offset += remote_row_size;
    }

    /* prepare the scatter/gather entry */
    sge[i * 2].addr = (uintptr_t) spdk_ptr_0 + i * length;
    sge[i * 2].length = length;
    sge[i * 2].lkey = res->mr->lkey;
    sge[i * 2 + 1].addr = (uintptr_t) spdk_ptr_1 + i * length;
    sge[i * 2 + 1].length = length;
    sge[i * 2 + 1].lkey = res->mr->lkey;

    /* prepare the send work request */
    sr[i].next = NULL;
    sr[i].wr_id = i;
    sr[i].sg_list = &sge[2 * i];
    sr[i].num_sge = 2;
    sr[i].opcode = opcode;
    sr[i].send_flags = IBV_SEND_SIGNALED;
    if (opcode!=IBV_WR_SEND) {
        sr[i].wr.rdma.remote_addr = res->remote_props.addr + remote_offset;
        sr[i].wr.rdma.rkey = res->remote_props.rkey;
    }

    /* there is a Receive Request in the responder side, so we won't get any into RNR flow */
    rc = ibv_post_send(res->qp, sr, &bad_wr);
    if (rc) {
        fprintf(stderr, "failed to post SR, rc: %d\n", rc);
        fprintf(stderr, "bad_wr->wr_id = %lu\n", bad_wr->wr_id);
    }
    else {
#if VERBOSE==1
        switch (opcode)
        {
        case IBV_WR_SEND:
            fprintf(stdout, "Send Request was posted\n");
            break;
        case IBV_WR_RDMA_READ:
            fprintf(stdout, "RDMA Read Request was posted\n");
            break;
        case IBV_WR_RDMA_WRITE:
            fprintf(stdout, "RDMA Write Request was posted\n");
            break;
        default:
            fprintf(stdout, "Unknown Request was posted\n");
            break;
        }
#endif
    }
    free(sr);
    free(sge);
    return rc;
}

/******************************************************************************
 * Function: post_send_multi_sr
 *
 * Input
 * res pointer to resources structure
 * opcode IBV_WR_SEND, IBV_WR_RDMA_READ or IBV_WR_RDMA_WRITE
 *
 * Output
 * none
 *
 * Returns
 * 0 on success, error code on failure
 *
 * Description
 * This function will create and post a send work request
 ******************************************************************************/
int post_send_multi_sr(struct resources *res, int opcode, char *spdk_ptr, uint64_t remote_offset, uint64_t rows, uint64_t remote_row_size, uint32_t length) {
    struct ibv_send_wr *sr;
    struct ibv_sge *sge;
    struct ibv_send_wr *bad_wr = NULL;
    int rc;
    uint64_t i;

    sr = calloc(rows, sizeof(struct ibv_send_wr));
    sge = calloc(rows, sizeof(struct ibv_sge));
    for (i = 0; i < rows - 1; i++) {
        /* prepare the scatter/gather entry */
        sge[i].addr = (uintptr_t) spdk_ptr + i * length;
        sge[i].length = length;
        sge[i].lkey = res->mr->lkey;

        /* prepare the send work request */
        sr[i].next = &sr[i+1];
        sr[i].wr_id = i;
        sr[i].sg_list = &sge[i];
        sr[i].num_sge = 1;
        sr[i].opcode = opcode;
        sr[i].send_flags = IBV_SEND_SIGNALED;
        if (opcode!=IBV_WR_SEND) {
            sr[i].wr.rdma.remote_addr = res->remote_props.addr + remote_offset;
            sr[i].wr.rdma.rkey = res->remote_props.rkey;
        }
        remote_offset += remote_row_size;
    }

    /* prepare the scatter/gather entry */
    sge[i].addr = (uintptr_t) spdk_ptr + i * length;
    sge[i].length = length;
    sge[i].lkey = res->mr->lkey;

    /* prepare the send work request */
    sr[i].next = NULL;
    sr[i].wr_id = i;
    sr[i].sg_list = &sge[i];
    sr[i].num_sge = 1;
    sr[i].opcode = opcode;
    sr[i].send_flags = IBV_SEND_SIGNALED;
    if (opcode!=IBV_WR_SEND) {
        sr[i].wr.rdma.remote_addr = res->remote_props.addr + remote_offset;
        sr[i].wr.rdma.rkey = res->remote_props.rkey;
    }
    remote_offset += remote_row_size;

    /* there is a Receive Request in the responder side, so we won't get any into RNR flow */
    rc = ibv_post_send(res->qp, sr, &bad_wr);
    if (rc)
        fprintf(stderr, "failed to post SR\n");
    else {
#if VERBOSE==1
        switch (opcode)
        {
        case IBV_WR_SEND:
            fprintf(stdout, "Send Request was posted\n");
            break;
        case IBV_WR_RDMA_READ:
            fprintf(stdout, "RDMA Read Request was posted\n");
            break;
        case IBV_WR_RDMA_WRITE:
            fprintf(stdout, "RDMA Write Request was posted\n");
            break;
        default:
            fprintf(stdout, "Unknown Request was posted\n");
            break;
        }
#endif
    }
    free(sr);
    free(sge);
    return rc;
}

/******************************************************************************
 * Function: post_send
 *
 * Input
 * res pointer to resources structure
 * opcode IBV_WR_SEND, IBV_WR_RDMA_READ or IBV_WR_RDMA_WRITE
 *
 * Output
 * none
 *
 * Returns
 * 0 on success, error code on failure
 *
 * Description
 * This function will create and post a send work request
 ******************************************************************************/
int post_send(struct resources *res, int opcode, char *spdk_ptr, uint64_t remote_offset, uint32_t length) {
    struct ibv_send_wr sr;
    struct ibv_sge sge;
    struct ibv_send_wr *bad_wr = NULL;
    int rc;
    /* prepare the scatter/gather entry */
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t) spdk_ptr;
    sge.length = length;
    sge.lkey = res->mr->lkey;
    /* prepare the send work request */
    memset(&sr, 0, sizeof(sr));
    sr.next = NULL;
    sr.wr_id = 0;
    sr.sg_list = &sge;
    sr.num_sge = 1;
    sr.opcode = opcode;
    sr.send_flags = IBV_SEND_SIGNALED;
    if (opcode!=IBV_WR_SEND) {
        sr.wr.rdma.remote_addr = res->remote_props.addr + remote_offset;
        sr.wr.rdma.rkey = res->remote_props.rkey;
    }
    /* there is a Receive Request in the responder side, so we won't get any into RNR flow */
    rc = ibv_post_send(res->qp, &sr, &bad_wr);
    if (rc)
        fprintf(stderr, "failed to post SR\n");
    else {
#if VERBOSE==1
        switch (opcode)
        {
        case IBV_WR_SEND:
            fprintf(stdout, "Send Request was posted\n");
            break;
        case IBV_WR_RDMA_READ:
            fprintf(stdout, "RDMA Read Request was posted\n");
            break;
        case IBV_WR_RDMA_WRITE:
            fprintf(stdout, "RDMA Write Request was posted\n");
            break;
        default:
            fprintf(stdout, "Unknown Request was posted\n");
            break;
        }
#endif
    }
    return rc;
}

/******************************************************************************
 * Function: post_receive
 *
 * Input
 * res pointer to resources structure
 *
 * Output
 * none
 *
 * Returns
 * 0 on success, error code on failure
 *
 * Description
 *
 ******************************************************************************/
int post_receive(struct resources *res) {
    struct ibv_recv_wr rr;
    struct ibv_sge sge;
    struct ibv_recv_wr *bad_wr;
    int rc;
    /* prepare the scatter/gather entry */
    memset(&sge, 0, sizeof(sge));
    sge.addr = (uintptr_t) res->buf;
    sge.length = LENGTH;
    sge.lkey = res->mr->lkey;
    /* prepare the receive work request */
    memset(&rr, 0, sizeof(rr));
    rr.next = NULL;
    rr.wr_id = 0;
    rr.sg_list = &sge;
    rr.num_sge = 1;
    /* post the Receive Request to the RQ */
    rc = ibv_post_recv(res->qp, &rr, &bad_wr);
    if (rc)
        fprintf(stderr, "failed to post RR\n");
    else
        fprintf(stdout, "Receive Request was posted\n");
    return rc;
}
