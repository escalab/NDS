#!/bin/bash
clean_cache="free && sync && echo 3 > /proc/sys/vm/drop_caches && free"
# prog_arr=("cublas_perftest_3" "cublas_perftest_5")
prog_arr=("cublas_perftest_5" "cublas_perftest_7" "cublas_perftest_9")
matrix_size=32768
submatrix_size=16384
end_size=32
iter_num=1

set -e 

if [ "$#" -ge 1 ]; then
    matrix_size=$1
fi

data=../../data/output_double_${matrix_size}
for prog in "${prog_arr[@]}"
do
    rm -f out_subsize_${prog}_${matrix_size}.txt
done

for ((pow=${submatrix_size}; pow >= ${end_size}; pow /= 2))
do
    for prog in "${prog_arr[@]}"
    do 
        echo ./${prog} ${data}_A.bin ${data}_B.bin ${matrix_size} ${pow} |& tee -a out_subsize_${prog}_${matrix_size}.txt
        for i in $(seq 1 ${iter_num})
        do
            ./${prog} ${data}_A.bin ${data}_B.bin ${matrix_size} ${pow} |& tee -a out_subsize_${prog}_${matrix_size}.txt
        done
    done
done
# echo "running program..."
# if [[ $EUID -eq 0 ]]; then
#     echo $clean_cache
#     ${clean_cache}
# fi
# if [ ${algo} -ge 2 ]; then
#     echo ./cublas_perftest_${algo} ${data}_A.bin ${data}_B.bin ${matrix_size} ${submatrix_size} 
#     ./cublas_perftest_${algo} ${data}_A.bin ${data}_B.bin ${matrix_size} ${submatrix_size} 
# else
#     echo ./cublas_perftest_${algo} ${data}_A.bin ${data}_B.bin ${matrix_size} 
#     ./cublas_perftest_${algo} ${data}_A.bin ${data}_B.bin ${matrix_size} 
# fi
