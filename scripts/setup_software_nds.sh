#!/bin/bash

# close
/home/yuchialiu/workspace/TensorStore/apps/hardware_close_device/close 19871
sleep 5

echo "going to switch to software configuration!"
sleep 5
/home/yuchialiu/workspace/spdk/examples/tensorstore/nds_software_async_req/nds -g -c ~/workspace/spdk/etc/spdk/nvmf_host.conf -x 5 -y 19871
/home/yuchialiu/workspace/spdk/examples/tensorstore/nds_software_async_req/nds -g -c ~/workspace/spdk/etc/spdk/nvmf_host.conf -x 5 -y 19871
/home/yuchialiu/workspace/spdk/examples/tensorstore/nds_software_async_req/nds -g -c ~/workspace/spdk/etc/spdk/nvmf_host.conf -x 5 -y 19871