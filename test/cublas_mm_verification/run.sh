#!/bin/bash
PROG='./cublas_gemm_verification'
MATRIX_SIZE=$1
SUBMATRIX_SIZE=$2
NEED_OUTPUT=$3

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <matrix_size> <submatrix_size> [need_output?]"
    exit 1
fi

COMMAND="${PROG} ../../data/output_double_${MATRIX_SIZE}.bin ../../data/output_double_block_${MATRIX_SIZE}_${SUBMATRIX_SIZE}.bin ${MATRIX_SIZE} ${SUBMATRIX_SIZE} ${NEED_OUTPUT}"
echo ${COMMAND}
${COMMAND}