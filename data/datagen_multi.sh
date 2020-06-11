#!/bin/bash
clean_cache="free && sync && echo 3 > /proc/sys/vm/drop_caches && free"
prog="datagen_double"

matrix_size=32768
submatrix_size=16384
end_size=2

set -e 

for ((pow=${submatrix_size}; pow >= ${end_size}; pow /= 2))
do
    echo ./${prog} output_double_${matrix_size}.bin output_double_block_${matrix_size}_${pow}.bin ${matrix_size} ${pow}
    ./${prog} output_double_${matrix_size}.bin output_double_block_${matrix_size}_${pow}.bin ${matrix_size} ${pow}
done
