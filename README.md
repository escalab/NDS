# NDS

This is the source code used for NDS: N-Dimensional Storage

[![DOI](https://zenodo.org/badge/404414517.svg)](https://zenodo.org/badge/latestdoi/404414517)

## Install
```
make
```

## Structure

* `cuda_samples`: Copied from cuda samples that helps to learn TCU wmma

* `data`: Contains matrix generator in sequential/NDS format

* `lib`:
  * `gemm`: offers APIs binding cublasGemm with NDS format
  * `spdkrpc`: communicates to SPDK app to fetch matrix from remote server
  * `tensorstore`: converts data format between sequential and NDS formats

* test:
  * `cublas_mm_verification`: Verify the resulting matrix is correct by using cublasGEMM
  * `matrix_transpose`: Verify and check the performance of fetching data from SPDK transposing matrix.
  * `mm_verification`: Verify the resulting matrix is correct by using blockmm on CPU
  * `read_test`: Verify tensorstore functionalities
  * `tensorstore_mm_perftest`: Performance test of APIs from gemm library
  * `tensorstore_mm_verification`: Verifications of APIs from gemm library
  * `transpose_perftest`: Performance test for cublasGEMM after applying normal/transposed matrix operands
