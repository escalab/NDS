#!/bin/bash
export PATH=$PATH:/usr/local/cuda/bin

script_dir="/home/yuchialiu/workspace/TensorStore/scripts"
nds_log="${script_dir}/software_nds.txt"

rm ${nds_log}

# block-GEMM
echo "block-GEMM"
cd ~/workspace/TensorStore/apps/software_block_gemm
make clean && make
./block_gemm 5 65536 16384 19871 >> ${nds_log}

# kmeans
echo "kmeans"
cd ~/workspace/TensorStore/apps/software_kmeans
make clean && make
./kmeans 6 65536 65536 4 19871 >> ${nds_log}

# knn
echo "knn"
cd ~/workspace/TensorStore/apps/software_knn
make clean && make
./knn ~/workspace/TensorStore/data/data/query_data_4096_65536.bin 6 65536 2048 128 4 19871 >> ${nds_log}

# bfs
echo "bfs"
cd ~/workspace/TensorStore/apps/software_bfs
make clean && make
./bfs 7 65536 4096 19871 >> ${nds_log}

# bellman-ford
echo "bellman-ford"
cd ~/workspace/TensorStore/apps/software_bellman_ford
make clean && make
./bellmanford 7 65536 4096 19871 >> ${nds_log}

# pagerank
echo "pagerank"
cd ~/workspace/TensorStore/apps/software_pagerank
make clean && make
./pagerank 8 65536 4096 19871 >> ${nds_log}

# convolution
echo "convolution"
cd ~/workspace/TensorStore/apps/software_convolution
make clean && make
./convolution 5 65536 4096 19871 >> ${nds_log}

# hotspot
echo "hotspot"
cd ~/workspace/TensorStore/apps/software_hotspot
make clean && make
./hotspot 5 65536 4096 19871 >> ${nds_log}

# tensor times vector
echo "tensor-times-vector"
cd ~/workspace/TensorStore/apps/software_ttv
make clean && make
./ttv 9 2048 512 19871 >> ${nds_log}

# tensor contraction
echo "tensor contraction"
cd ~/workspace/TensorStore/apps/software_tc
make clean && make
./tc 9 2048 512 19871 >> ${nds_log}

# close software NDS server
echo "close software NDS server"
cd ~/workspace/TensorStore/apps/software_close_device
make clean && make
./close 19871

cd ~/workspace/TensorStore/scripts/
