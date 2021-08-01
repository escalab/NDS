#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: ./run.sh <m> <sub_m>"
    echo "m = # of points of the input"
    echo "sub_m = one dimension size of the compute kernel"
    exit 1
fi

echo "./knn ~/TensorStore/data/data/query_data_4096_65536.bin 6 $1 $2 128 4 19871"
./knn ~/TensorStore/data/data/query_data_4096_65536.bin 6 $1 $2 128 4 19871