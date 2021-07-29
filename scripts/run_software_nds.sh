#!/bin/bash
export PATH=$PATH:/usr/local/cuda/bin

script_dir="/home/yuchialiu/workspace/TensorStore/scripts"
seq_log="${script_dir}/software_seq.txt"
nds_log="${script_dir}/software_nds.txt"

echo "touching ${seq_log}..."
touch ${seq_log}

echo "touching ${nds_log}..."
touch ${nds_log}
# block-GEMM (doesn't work somehow)
echo "block-GEMM"
cd ~/workspace/TensorStore/apps/software_block_gemm
make clean && make
./block_gemm 0 65536 16384 19871 | tee -a ${seq_log}
./block_gemm 5 65536 16384 19871 | tee -a ${nds_log}

# kmeans
echo "kmeans"
cd ~/workspace/TensorStore/apps/software_kmeans
make clean && make
./kmeans 1 65536 65536 4 19871 | tee -a ${seq_log}
./kmeans 6 65536 65536 4 19871 | tee -a ${nds_log}

# knn
echo "knn"
cd ~/workspace/TensorStore/apps/software_knn
make clean && make
./knn query_data_4096_65536.bin 1 65536 2048 128 4 19871 | tee -a ${seq_log}
./knn query_data_4096_65536.bin 6 65536 2048 128 4 19871 | tee -a ${nds_log}

# bfs
echo "bfs"
cd ~/workspace/TensorStore/apps/software_bfs
make clean && make
./bfs 2 65536 4096 19871 | tee -a ${seq_log}
./bfs 7 65536 4096 19871 | tee -a ${nds_log}

# bellman-ford
echo "bellman-ford"
cd ~/workspace/TensorStore/apps/software_bellman_ford
make clean && make
./bellmanford 2 65536 4096 19871 | tee -a ${seq_log}
./bellmanford 7 65536 4096 19871 | tee -a ${nds_log}

# pagerank
echo "pagerank"
cd ~/workspace/TensorStore/apps/software_pagerank
make clean && make
./pagerank 3 65536 4096 19871 | tee -a ${seq_log}
./pagerank 8 65536 4096 19871 | tee -a ${nds_log}

# convolution
echo "convolution"
cd ~/workspace/TensorStore/apps/software_convolution
make clean && make
./convolution 0 65536 4096 19871 | tee -a ${seq_log}
./convolution 5 65536 4096 19871 | tee -a ${nds_log}

# hotspot
echo "hotspot"
cd ~/workspace/TensorStore/apps/software_hotspot
make clean && make
./hotspot 0 65536 4096 19871 | tee -a ${seq_log}
./hotspot 5 65536 4096 19871 | tee -a ${nds_log}

# tensor times vector
echo "tensor-times-vector"
cd ~/workspace/TensorStore/apps/software_ttv
make clean && make
./ttv 4 2048 512 19871 | tee -a ${seq_log}
./ttv 9 2048 512 19871 | tee -a ${nds_log}

# tensor contraction
echo "tensor contraction"
cd ~/workspace/TensorStore/apps/software_tc
make clean && make
./tc 4 2048 512 19871 | tee -a ${seq_log}
./tc 9 2048 512 19871 | tee -a ${nds_log}

# close software NDS server
echo "close software NDS server"
cd ~/workspace/TensorStore/apps/software_close_device
make clean && make
./close 19871

cd ~/workspace/TensorStore/scripts/
