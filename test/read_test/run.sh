#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <matrix_size> <submatrix_size>"
fi

for i in {0..5}
do
    free && sync && echo 3 > /proc/sys/vm/drop_caches && free
    ./read_sequential output.bin $1 $2
done

for i in {0..5}
do
    free && sync && echo 3 > /proc/sys/vm/drop_caches && free
    ./read_block output_block.bin $1 $2
done
