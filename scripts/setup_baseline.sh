#!/bin/bash

set -e

# setup shared buffer
truncate -s 4G /dev/hugepages/tensorstore
chmod 777 /dev/hugepages/tensorstore

# send command to SST-100 to invoke SPDK NVMe-oF


# send command to host daemon to invome host SPDK service
~/workspace/spdk/examples/tensorstore/nds_software_async_req/nds -g -c ~/workspace/spdk/etc/spdk/nvmf_host.conf -x 5 -y 19871 &