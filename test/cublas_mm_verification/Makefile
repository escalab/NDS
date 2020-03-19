NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include  -lcublas_static -lculibos -lcudart_static -lpthread -ldl -lcublasLt_static
LD_FLAGS    = -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcublas_static -lculibos -lcudart_static -lpthread -ldl -lcublasLt_static

LIB=libcublasGEMM.so


ifeq ($(debug),1)
	LD_FLAGS += -g -DDEBUG
endif

default: all

all: lib cublas_gemm_verification cublas_perftest

lib: cublasGEMM.cu
	$(NVCC) --compiler-options '-fPIC' -o $(LIB) --shared cublasGEMM.cu $(NVCC_FLAGS)

cublas_gemm_verification: verify.cu
	$(NVCC) -o $@ $< $(LIB) $(LD_FLAGS) 

cublas_perftest: perftest.cu
	$(NVCC) -o $@ $< $(LIB) $(LD_FLAGS) 

clean:
	rm -rf cublas_perftest cublas_gemm_verification $(LIB)

clean_answer:
	rm -rf *.bin
