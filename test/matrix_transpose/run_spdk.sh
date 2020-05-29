#!/bin/bash
clean_cache="free && sync && echo 3 > /proc/sys/vm/drop_caches && free"
prog="test_transpose_spdk"
output="output_spdk_tensor.txt"

matrix_size=32768
submatrix_size=8192
end_size=64
iter_num=1
id=0
set -e 

rm -rf ${output} && touch ${output}

for ((pow=${submatrix_size}; pow >= ${end_size}; pow /= 2, id+=1))
do
    command="./${prog} ${id} ${matrix_size} ${pow} ../../data/output_double_${matrix_size}.bin"
    echo ${command} |& tee -a ${output}
    for i in $(seq 1 ${iter_num})
    do
        free && sync && echo 3 > /proc/sys/vm/drop_caches && free
        ${command} |& tee -a ${output}
    done
done
