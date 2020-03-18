#!/bin/bash
PROG='./cudaTensorCoreGemm'
MATRIX_SIZE=$1
SUBMATRIX_SIZE=$2

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <matrix_size> <submatrix_size>"
    exit 1
fi

echo ${PROG} ../../../data/output_double_block_${MATRIX_SIZE}_${SUBMATRIX_SIZE}.bin ${MATRIX_SIZE} ${MATRIX_SIZE} ${MATRIX_SIZE} ../../../test/cublas_mm_verification/ans_block_${MATRIX_SIZE}_${SUBMATRIX_SIZE}.bin
${PROG} ../../../data/output_double_block_${MATRIX_SIZE}_${SUBMATRIX_SIZE}.bin ${MATRIX_SIZE} ${MATRIX_SIZE} ${MATRIX_SIZE} ../../../test/cublas_mm_verification/ans_block_${MATRIX_SIZE}_${SUBMATRIX_SIZE}.bin