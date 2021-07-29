#!/bin/bash

# download graph from website
wget https://www.cc.gatech.edu/dimacs10/archive/data/kronecker/kron_g500-simple-logn16.graph.bz2
bzip2 -d kron_g500-simple-logn16.graph.bz2

# generate binary adjacency matrix from ASCII adjacency list
python3 graph_txt2bin.py kron_g500-simple-logn16.graph float64
