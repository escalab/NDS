#!/bin/bash
clean_cache="free && sync && echo 3 > /proc/sys/vm/drop_caches && free"

algo=0
matrix_size=32768
submatrix_size=16384

data=../../data/output_double_${matrix_size}
tensor_data=../../data/output_double_block_${matrix_size}_${submatrix_size}


if [ "$#" -ge 1 ]; then
    algo=$1
else
    echo "usage $0 <algorithm>"
    exit 1
fi

if [ ${algo} -ge 4 ]; then
    data=${tensor_data}
fi

echo "running program..."
if [[ $EUID -eq 0 ]]; then
    echo $clean_cache
    ${clean_cache}
fi
if [ ${algo} -ge 2 ]; then
    echo ./cublas_perftest_${algo} ${data}_A.bin ${data}_B.bin ${matrix_size} ${submatrix_size} 
    ./cublas_perftest_${algo} ${data}_A.bin ${data}_B.bin ${matrix_size} ${submatrix_size} 
else
    echo ./cublas_perftest_${algo} ${data}_A.bin ${data}_B.bin ${matrix_size} 
    ./cublas_perftest_${algo} ${data}_A.bin ${data}_B.bin ${matrix_size} 
fi
