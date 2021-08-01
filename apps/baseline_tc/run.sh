#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: ./run.sh <m> <sub_m>"
    echo "m = # of points of the input"
    echo "sub_m = one dimension size of the compute kernel"
    exit 1
fi

echo "./tc 4 $1 $2 19871"
./tc 4 $1 $2 19871