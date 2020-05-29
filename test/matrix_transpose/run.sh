#!/bin/bash
clean_cache="free && sync && echo 3 > /proc/sys/vm/drop_caches && free"
prog="test_transpose"

matrix_size=32768
submatrix_size=16384
end_size=2
iter_num=1

set -e 

rm -rf output.txt && touch output.txt

for ((pow=${submatrix_size}; pow >= ${end_size}; pow /= 2))
do
    command="./${prog} ${matrix_size} ${pow} ../../data/output_double_${matrix_size}.bin ../../data/output_double_block_${matrix_size}_${pow}.bin"
    echo ${command} |& tee -a output.txt
    for i in $(seq 1 ${iter_num})
    do
        free && sync && echo 3 > /proc/sys/vm/drop_caches && free
        ${command} |& tee -a output.txt
    done
done
