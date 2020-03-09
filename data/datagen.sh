#!/bin/bash

MATRIX_SIZE=$1
SUBMATRIX_SIZE=$2
TYPE="long"

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <matrix_size> <submatrix_size> <double_type?>"
    exit 1
fi

if [[ $3 -eq 0 ]] ; then
    TYPE="long"
else
    TYPE="double"
fi

echo ${MATRIX_SIZE} ${SUBMATRIX_SIZE} ${TYPE}
./datagen_${TYPE} output_${TYPE}_${MATRIX_SIZE}.bin output_${TYPE}_block_${MATRIX_SIZE}.bin ${MATRIX_SIZE} ${SUBMATRIX_SIZE}
