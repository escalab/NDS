NVCC = nvcc
LD_FLAGS = -O3 -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcublas_static -lculibos -lcudart_static -lpthread -ldl -lcublasLt_static
LD_FLAGS += -I../../lib/gemm -L../../lib/gemm -lcublasGEMM

TARGET=cublas_gemm_verification_0 cublas_gemm_verification_1 cublas_gemm_verification_2 cublas_gemm_verification_3 cublas_gemm_verification_4 
TARGET+=cublas_gemm_verification_5 cublas_gemm_verification_6 cublas_gemm_verification_7 cublas_gemm_verification_8 cublas_gemm_verification_9 cublas_gemm_verification_10
SRC=verify.cu


ifeq ($(debug),1)
	LD_FLAGS += -g -DDEBUG
endif

default: all

all: $(TARGET)

cublas_gemm_verification_0: $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DALGO=0

cublas_gemm_verification_1: $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DALGO=1

cublas_gemm_verification_2: $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DALGO=2

cublas_gemm_verification_3: $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DALGO=3

cublas_gemm_verification_4: $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DALGO=4

cublas_gemm_verification_5: $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DALGO=5

cublas_gemm_verification_6: $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DALGO=6

cublas_gemm_verification_7: $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DALGO=7

cublas_gemm_verification_8: $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DALGO=8

cublas_gemm_verification_9: $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DALGO=9

cublas_gemm_verification_10: $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DALGO=10

clean:
	rm -rf $(TARGET)

clean_answer:
	rm -rf *.bin
