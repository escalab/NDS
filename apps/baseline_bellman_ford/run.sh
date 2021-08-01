#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: ./run.sh <m> <sub_m>"
    echo "m = one dimension size of the input"
    echo "sub_m = one dimension size of the compute kernel"
    exit 1
fi

echo "./bellmanford 2 65536 4096 19871"
./bellmanford 2 $1 $2 19871