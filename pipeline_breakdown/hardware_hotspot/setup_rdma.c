//
// Created by Hank Liu on 2019-05-13.
//

#include "rdma.h"

int modify_qp_to_init(struct ibv_qp *qp, struct config_t *config);
int modify_qp_to_rtr(struct ibv_qp *qp, uint32_t remote_qpn, uint16_t dlid, uint8_t *dgid, struct config_t *config);
int modify_qp_to_rts(struct ibv_qp *qp);

void config_destroy(struct config_t *config) {
    if (config->dev_name!=NULL) {
        free((void *) config->dev_name);
    }
}

/******************************************************************************
 * Function: modify_qp_to_init
 *
 * Input
 * qp QP to transition
 *
 * Output
 * none
 *
 * Returns
 * 0 on success, ibv_modify_qp failure code on failure
 *
 * Description
 * Transition a QP from the RESET to INIT state
 ******************************************************************************/
int modify_qp_to_init(struct ibv_qp *qp, struct config_t *config) {
    struct ibv_qp_attr attr;
    int flags;
    int rc;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = config->ib_port;
    attr.pkey_index = 0;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE;
    flags =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    rc = ibv_modify_qp(qp, &attr, flags);
    if (rc)
        fprintf(stderr, "failed to modify QP state to INIT\n");
    return rc;
}

/******************************************************************************
 * Function: modify_qp_to_rtr
 *
 * Input
 * qp QP to transition
 * remote_qpn remote QP number
 * dlid destination LID
 * dgid destination GID (mandatory for RoCEE)
 *
 * Output
 * none
 *
 * Returns
 * 0 on success, ibv_modify_qp failure code on failure
 *
 * Description
 * Transition a QP from the INIT to RTR state, using the specified QP number
 ******************************************************************************/
int modify_qp_to_rtr(struct ibv_qp *qp, uint32_t remote_qpn, uint16_t dlid,
                     uint8_t *dgid, struct config_t *config) {
    struct ibv_qp_attr attr;
    int flags;
    int rc;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_4096;
    attr.dest_qp_num = remote_qpn;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 0x12;
    attr.ah_attr.is_global = 0;
    attr.ah_attr.dlid = dlid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = config->ib_port;
    if (config->gid_idx >= 0) {
        attr.ah_attr.is_global = 1;
        attr.ah_attr.port_num = 1;
        memcpy(&attr.ah_attr.grh.dgid, dgid, 16);
        attr.ah_attr.grh.flow_label = 0;
        attr.ah_attr.grh.hop_limit = 1;
        attr.ah_attr.grh.sgid_index = config->gid_idx;
        attr.ah_attr.grh.traffic_class = 0;
    }
    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    rc = ibv_modify_qp(qp, &attr, flags);
    if (rc)
        fprintf(stderr, "failed to modify QP state to RTR, return code: %d\n", rc);
    return rc;
}

/******************************************************************************
 * Function: modify_qp_to_rts
 *
 * Input
 * qp QP to transition
 *
 * Output
 * none
 *
 * Returns
 * 0 on success, ibv_modify_qp failure code on failure
 *
 * Description
 * Transition a QP from the RTR to RTS state
 ******************************************************************************/
int modify_qp_to_rts(struct ibv_qp *qp) {
    struct ibv_qp_attr attr;
    int flags;
    int rc;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 0x12;
    attr.retry_cnt = 6;
    attr.rnr_retry = 0;
    attr.sq_psn = 0;
    attr.max_rd_atomic = 1;
    flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
        IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
    rc = ibv_modify_qp(qp, &attr, flags);
    if (rc)
        fprintf(stderr, "failed to modify QP state to RTS\n");
    return rc;
}

/******************************************************************************
 * Function: connect_qp
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
 * Connect the QP. Transition the server side to RTR, sender side to RTS
 ******************************************************************************/
int connect_qp(struct resources *res, struct config_t *config) {
    struct cm_con_data_t local_con_data;
    struct cm_con_data_t remote_con_data;
    struct cm_con_data_t tmp_con_data;
    int rc = 0;
    char temp_char;
    union ibv_gid my_gid;
    if (config->gid_idx >= 0) {
        rc =
            ibv_query_gid(res->ib_ctx, config->ib_port, config->gid_idx, &my_gid);
        if (rc) {
            fprintf(stderr, "could not get gid for port %d, index %d\n",
                    config->ib_port, config->gid_idx);
            return rc;
        }
    } else
        memset(&my_gid, 0, sizeof my_gid);
    /* exchange using TCP sockets info required to connect QPs */
    local_con_data.addr = htonll((uintptr_t) res->buf);
    local_con_data.rkey = htonl(res->mr->rkey);
    local_con_data.qp_num = htonl(res->qp->qp_num);
    local_con_data.lid = htons(res->port_attr.lid);
    memcpy(local_con_data.gid, &my_gid, 16);
#if VERBOSE==1
    fprintf(stdout, "\nLocal LID = 0x%x\n", res->port_attr.lid);
#endif
    if (sock_sync_data(res->sock, sizeof(struct cm_con_data_t), (char *) &local_con_data,
                       (char *) &tmp_con_data) < 0) {
        fprintf(stderr, "failed to exchange connection data between sides\n");
        rc = 1;
        goto connect_qp_exit;
    }
    remote_con_data.addr = ntohll(tmp_con_data.addr);
    remote_con_data.rkey = ntohl(tmp_con_data.rkey);
    remote_con_data.qp_num = ntohl(tmp_con_data.qp_num);
    remote_con_data.lid = ntohs(tmp_con_data.lid);
    memcpy(remote_con_data.gid, tmp_con_data.gid, 16);
    /* save the remote side attributes, we will need it for the post SR */
    res->remote_props = remote_con_data;
#if VERBOSE==1
    fprintf(stdout, "Remote address = 0x%" PRIx64 "\n", remote_con_data.addr);
    fprintf(stdout, "Remote rkey = 0x%x\n", remote_con_data.rkey);
    fprintf(stdout, "Remote QP number = 0x%x\n", remote_con_data.qp_num);
    fprintf(stdout, "Remote LID = 0x%x\n", remote_con_data.lid);
    if (config->gid_idx >= 0)
    {
        uint8_t *p = remote_con_data.gid;
        fprintf(stdout,
                "Remote GID = %02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x\n",
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
                p[10], p[11], p[12], p[13], p[14], p[15]);
    }
#endif
    /* modify the QP to init */
    rc = modify_qp_to_init(res->qp, config);
    if (rc) {
        fprintf(stderr, "change QP state to INIT failed\n");
        goto connect_qp_exit;
    }
    /* modify the QP to RTR */
    rc =
        modify_qp_to_rtr(res->qp, remote_con_data.qp_num, remote_con_data.lid,
                         remote_con_data.gid, config);
    if (rc) {
        fprintf(stderr, "failed to modify QP state to RTR\n");
        goto connect_qp_exit;
    }
#if VERBOSE==1
    fprintf(stdout, "Modified QP state to RTR\n");
#endif
    rc = modify_qp_to_rts(res->qp);
    if (rc) {
        fprintf(stderr, "failed to modify QP state to RTR\n");
        goto connect_qp_exit;
    }
#if VERBOSE==1
    fprintf(stdout, "QP state was change to RTS\n\n");
#endif
    /* sync to make sure that both sides are in states that they can connect to prevent packet loose */
    if (sock_sync_data(res->sock, 1, "Q", &temp_char)) /* just send a dummy char back and forth */
    {
        fprintf(stderr, "sync error after QPs are were moved to RTS\n");
        rc = 1;
    }
connect_qp_exit:
    return rc;
}

/******************************************************************************
 * Function: resources_destroy
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
 * Cleanup and deallocate all resources used
 ******************************************************************************/
int resources_destroy(struct resources *res) {
    int rc = 0;
    if (res->qp) {
        if (ibv_destroy_qp(res->qp)) {
            fprintf(stderr, "failed to destroy QP\n");
            rc = 1;
        }
        res->qp = NULL;
    }

    if (res->mr) {
        if (ibv_dereg_mr(res->mr)) {
            fprintf(stderr, "failed to deregister MR\n");
            rc = 1;
        }
        res->mr = NULL;
    }

    if (res->cq) {
        if (ibv_destroy_cq(res->cq)) {
            fprintf(stderr, "failed to destroy CQ\n");
            rc = 1;
        }
        res->cq = NULL;
    }

    if (res->pd) {
        if (ibv_dealloc_pd(res->pd)) {
            fprintf(stderr, "failed to deallocate PD\n");
            rc = 1;
        }
        res->pd = NULL;
    }

    if (res->ib_ctx) {
        if (ibv_close_device(res->ib_ctx)) {
            fprintf(stderr, "failed to close device context\n");
            rc = 1;
        }
        res->ib_ctx = NULL;
    }

    if (res->sock >= 0) {
        if (close(res->sock)) {
            fprintf(stderr, "failed to close socket\n");
            rc = 1;
        }
        res->sock = -1;
    }

    return rc;
}

/******************************************************************************
 * Function: resources_init
 *
 * Input
 * res pointer to resources structure
 *
 * Output
 * res is initialized
 *
 * Returns
 * none
 *
 * Description
 * res is initialized to default values
 ******************************************************************************/
void resources_init(struct resources *res) {
    memset(res, 0, sizeof(struct resources));
    res->sock = -1;
}

int resources_find_device(struct resources *res, struct config_t *config) {
    struct ibv_device **dev_list = NULL;
    struct ibv_device *ib_dev = NULL;
    int i;
    int num_devices;
    int rc = 0;
#if VERBOSE==1
    fprintf(stdout, "searching for IB devices in host\n");
#endif
    /* get device names in the system */
    dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list) {
        fprintf(stderr, "failed to get IB devices list\n");
        rc = 1;
        resources_destroy(res);
        goto resources_find_device_exit;
    }
    /* if there isn't any IB device in host */
    if (!num_devices) {
        fprintf(stderr, "found %d device(s)\n", num_devices);
        rc = 1;
        resources_destroy(res);
        goto resources_find_device_exit;
    }
#if VERBOSE==1
    fprintf(stdout, "found %d device(s)\n", num_devices);
#endif
    /* search for the specific device we want to work with */
    for (i = 0; i < num_devices; i++) {
        if (!config->dev_name) {
            config->dev_name = strdup(ibv_get_device_name(dev_list[i]));
            fprintf(stdout,
                    "device not specified, using first one found: %s\n",
                    config->dev_name);
            for (int j = 0; j < num_devices; j++) {
                fprintf(stdout,
                        "device %d: %s\n", j, ibv_get_device_name(dev_list[j]));
            }
        }
        if (!strcmp(ibv_get_device_name(dev_list[i]), config->dev_name)) {
            ib_dev = dev_list[i];
            break;
        }
    }
    /* if the device wasn't found in host */
    if (!ib_dev) {
        fprintf(stderr, "IB device %s wasn't found\n", config->dev_name);
        rc = 1;
        resources_destroy(res);
        goto resources_find_device_exit;
    }
    /* get device handle */
    res->ib_ctx = ibv_open_device(ib_dev);
    if (!res->ib_ctx) {
        fprintf(stderr, "failed to open device %s\n", config->dev_name);
        rc = 1;
        resources_destroy(res);
        goto resources_find_device_exit;
    }
    /* We are now done with device list, free it */
    resources_find_device_exit:
    ibv_free_device_list(dev_list);
    return rc;
}

/******************************************************************************
 * Function: resources_create
 *
 * Input
 * res pointer to resources structure to be filled in
 *
 * Output
 * res filled in with resources
 *
 * Returns
 * 0 on success, 1 on failure
 *
 * Description
 *
 * This function creates and allocates all necessary system resources. These
 * are stored in res.
 *****************************************************************************/
int resources_create(struct resources *res, struct config_t *config) {
    struct ibv_qp_init_attr qp_init_attr;
    int mr_flags = 0;
    int cq_size = 0;
    int rc = 0;

    /* query port properties */
    if (ibv_query_port(res->ib_ctx, config->ib_port, &res->port_attr)) {
        fprintf(stderr, "ibv_query_port on port %u failed\n", config->ib_port);
        rc = 1;
        goto resources_create_exit;
    }
    /* allocate Protection Domain */
    res->pd = ibv_alloc_pd(res->ib_ctx);
    if (!res->pd) {
        fprintf(stderr, "ibv_alloc_pd failed\n");
        rc = 1;
        goto resources_create_exit;
    }
    /* each side will send only one WR, so Completion Queue with 1 entry is enough */
    cq_size = CQ_NUM_ENTRIES;
    res->cq = ibv_create_cq(res->ib_ctx, cq_size, NULL, NULL, 0);
    if (!res->cq) {
        fprintf(stderr, "failed to create CQ with %u entries\n", cq_size);
        rc = 1;
        goto resources_create_exit;
    }
    /* allocate the memory buffer that will hold the data */
    res->size = BUF_SIZE;
    if (res->buf==NULL) {
        // res->buf = (char *)memalign(4096, res->size);
        res->buf = (char *) malloc(res->size);
        if (!res->buf) {
            fprintf(stderr, "failed to malloc %Zu bytes to memory buffer\n", res->size);
            rc = 1;
            goto resources_create_exit;
        }
        memset(res->buf, 0, res->size);
    }

    /* register the memory buffer */
    printf("register memory region, size is %lu\n", res->size);
    mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE;
    res->mr = ibv_reg_mr(res->pd, res->buf, res->size, mr_flags);
    if (!res->mr) {
        fprintf(stderr, "ibv_reg_mr failed with mr_flags=0x%x\n", mr_flags);
        rc = 1;
        goto resources_create_exit;
    }
    fprintf(stdout,
            "MR was registered with addr=%p, lkey=0x%x, rkey=0x%x, flags=0x%x\n",
            res->buf, res->mr->lkey, res->mr->rkey, mr_flags);
    /* create the Queue Pair */
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.sq_sig_all = 0;
    qp_init_attr.send_cq = res->cq;
    qp_init_attr.recv_cq = res->cq;
    qp_init_attr.cap.max_send_wr = CQ_NUM_ENTRIES;
    qp_init_attr.cap.max_recv_wr = CQ_NUM_ENTRIES;
    qp_init_attr.cap.max_send_sge = 4;
    qp_init_attr.cap.max_recv_sge = 4;
    res->qp = ibv_create_qp(res->pd, &qp_init_attr);
    if (!res->qp) {
        fprintf(stderr, "failed to create QP\n");
        rc = 1;
        goto resources_create_exit;
    }
    fprintf(stdout, "QP was created, QP number=0x%x\n", res->qp->qp_num);

resources_create_exit:
    if (rc) {
        /* Error encountered, cleanup */
        if (res->qp) {
            ibv_destroy_qp(res->qp);
            res->qp = NULL;
        }
        if (res->mr) {
            ibv_dereg_mr(res->mr);
            res->mr = NULL;
        }
        if (res->buf) {
            free(res->buf);
            res->buf = NULL;
        }
        if (res->cq) {
            ibv_destroy_cq(res->cq);
            res->cq = NULL;
        }
        if (res->pd) {
            ibv_dealloc_pd(res->pd);
            res->pd = NULL;
        }
        if (res->ib_ctx) {
            ibv_close_device(res->ib_ctx);
            res->ib_ctx = NULL;
        }
        if (res->sock >= 0) {
            if (close(res->sock))
                fprintf(stderr, "failed to close socket\n");
            res->sock = -1;
        }
    }
    return rc;
}