#!/bin/bash
PROG='./cublas_gemm_verification'
MATRIX_SIZE=$1
SUBMATRIX_SIZE=$2

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <matrix_size> <submatrix_size>"
    exit 1
fi

COMMAND="${PROG} ../../data/output_double_${MATRIX_SIZE}.bin ../../data/output_double_block_${MATRIX_SIZE}_${SUBMATRIX_SIZE}.bin ${MATRIX_SIZE} ${SUBMATRIX_SIZE}"
echo ${COMMAND}
${COMMAND}