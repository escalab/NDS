#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: ./run.sh <m> <attributes>"
    echo "m = # of points of the input"
    echo "attributes = # of attributes of the input"
    exit 1
fi

echo "./kmeans 6 $1 $2 4 19871"
./kmeans 6 $1 $2 4 19871
