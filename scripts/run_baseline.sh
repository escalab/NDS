#!/bin/bash
export PATH=$PATH:/usr/local/cuda/bin

script_dir="${HOME}/TensorStore/scripts"
baseline_log="${script_dir}/baseline.txt"

rm ${baseline_log}

# block-GEMM
echo "block-GEMM"
cd ~/TensorStore/apps/baseline_block_gemm
make clean && make
./block_gemm 0 65536 8192 19871 >> ${baseline_log}

# kmeans
echo "kmeans"
cd ~/TensorStore/apps/baseline_kmeans
make clean && make
./kmeans 1 65536 65536 4 19871 >> ${baseline_log}

# knn
echo "knn"
cd ~/TensorStore/apps/baseline_knn
make clean && make
./knn ~/TensorStore/data/data/query_data_4096_65536.bin 1 65536 2048 128 4 19871 >> ${baseline_log}

# bfs
echo "bfs"
cd ~/TensorStore/apps/baseline_bfs
make clean && make
./bfs 2 65536 4096 19871 >> ${baseline_log}

# bellman-ford
echo "bellman-ford"
cd ~/TensorStore/apps/baseline_bellman_ford
make clean && make
./bellmanford 2 65536 4096 19871 >> ${baseline_log}

# pagerank
echo "pagerank"
cd ~/TensorStore/apps/baseline_pagerank
make clean && make
./pagerank 3 65536 4096 19871 >> ${baseline_log}

# convolution
echo "convolution"
cd ~/TensorStore/apps/baseline_convolution
make clean && make
./convolution 0 65536 4096 19871 >> ${baseline_log}

# hotspot
echo "hotspot"
cd ~/TensorStore/apps/baseline_hotspot
make clean && make
./hotspot 0 65536 4096 19871 >> ${baseline_log}

# tensor times vector
echo "tensor-times-vector"
cd ~/TensorStore/apps/baseline_ttv
make clean && make
./ttv 4 2048 512 19871 >> ${baseline_log}

# tensor contraction
echo "tensor contraction"
cd ~/TensorStore/apps/baseline_tc
make clean && make
./tc 4 2048 512 19871 >> ${baseline_log}

# close software NDS server
echo "close software NDS server"
cd ~/TensorStore/apps/baseline_close_device
make clean && make
./close 19871

cd ${script_dir}
