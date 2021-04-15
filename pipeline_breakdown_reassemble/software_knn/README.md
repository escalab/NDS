# how to run

```
free && sync && echo 3 > /proc/sys/vm/drop_caches && free && make && ./knn ~/workspace/kNN-CUDA/code/query_data_128_65536.bin 5 65536 2048 16 128 19877
```