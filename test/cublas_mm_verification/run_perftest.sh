#!/bin/bash

nvprof_path="/usr/local/cuda-10.2/bin/nvprof"
seq_prog_arr=("cublas_perftest_0" "cublas_perftest_1") 
block_seq=("cublas_perftest_2" "cublas_perftest_3")
tensor_prog_arr=("cublas_perftest_4" "cublas_perftest_5")

matrix_size=16384
submatrix_size=8192

matrix_data=../../data/output_double_${matrix_size}.bin
tensor_data=../../data/output_double_block_${matrix_size}_${submatrix_size}.bin

iter_num=1

if [ "$#" -ge 1 ]; then
    iter_num=$1
fi

for prog in "${seq_prog_arr[@]}"
do
    rm -f out_${prog}.txt
done

for prog in "${block_seq[@]}"
do
    rm -f out_${prog}.txt
done

for prog in "${tensor_prog_arr[@]}"
do
    rm -f out_${prog}.txt
done

echo "doing sequential GEMM"
for prog in "${seq_prog_arr[@]}"
do
    for i in $(seq 1 ${iter_num})
    do
        echo ./${prog} ${matrix_data} ${matrix_size}
        ./${prog} ${matrix_data} ${matrix_size} | tee -a out_${prog}.txt
        ${nvprof_path} ./${prog} ${matrix_data} ${matrix_size} | tee -a out_${prog}.txt
    done
done

echo "doing sequential block-GEMM"
for prog in "${block_seq[@]}"
do
    for i in $(seq 1 ${iter_num})
    do
        echo ./${prog} ${matrix_data} ${matrix_size} ${submatrix_size}
        ./${prog} ${matrix_data} ${matrix_size} ${submatrix_size} | tee -a out_${prog}.txt
        ${nvprof_path} ./${prog} ${matrix_data} ${matrix_size} ${submatrix_size} | tee -a out_${prog}.txt
    done
done

echo "doing tensor block-GEMM"
for prog in "${tensor_prog_arr[@]}"
do
    for i in $(seq 1 ${iter_num})
    do
        echo ./${prog} ${tensor_data} ${matrix_size} ${submatrix_size}
        ./${prog} ${tensor_data} ${matrix_size} ${submatrix_size} | tee -a out_${prog}.txt
        ${nvprof_path} ./${prog} ${tensor_data} ${matrix_size} ${submatrix_size} | tee -a out_${prog}.txt
    done
done
