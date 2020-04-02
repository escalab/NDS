NVCC = nvcc
LD_FLAGS = -O3 -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcublas_static -lculibos -lcudart_static -lpthread -ldl -lcublasLt_static
LD_FLAGS += -I../../lib/gemm -L../../lib/gemm -lcublasGEMM

SRC=perftest.cu
TARGET=cublas_perftest_0 cublas_perftest_1 cublas_perftest_2 cublas_perftest_3 cublas_perftest_4 
TARGET+=cublas_perftest_5 cublas_perftest_6

ifeq ($(debug),1)
	LD_FLAGS += -g -DDEBUG
endif

default: all

all: $(TARGET)

cublas_perftest_0: $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DALGO=0

cublas_perftest_1: $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DALGO=1

cublas_perftest_2: $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DIS_BLOCK_ALGO -DALGO=2

cublas_perftest_3: $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DIS_BLOCK_ALGO -DALGO=3

cublas_perftest_4: $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DIS_BLOCK_ALGO -DALGO=4

cublas_perftest_5: $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DIS_BLOCK_ALGO -DALGO=5

cublas_perftest_6: $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DIS_BLOCK_ALGO -DALGO=6

clean:
	rm -rf $(TARGET)