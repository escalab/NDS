#!/bin/bash

# compile generator
make

# generate reference data
./datagen ref_data_65536_65536.bin 65536 65536

# generate query data
./datagen query_data_4096_65536.bin 4096 65536
