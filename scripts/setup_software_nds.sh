#!/bin/bash

set -e

# send command to host daemon to invome host SPDK service
~/workspace/spdk/examples/tensorstore/nds_software_async_req/nds -g -c ~/workspace/spdk/etc/spdk/nvmf_host.conf -x 5 -y 19871 &