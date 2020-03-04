#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <matrix_size> <submatrix_size>"
fi

./read_sequential output.bin $1 $2
./read_block output_block.bin $1 $2