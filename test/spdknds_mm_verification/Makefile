NVCC = nvcc
LD_FLAGS = -O3 -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcublas_static -lculibos -lcudart_static -lpthread -ldl -lcublasLt_static
LD_FLAGS += -I../../lib/gemm -L../../lib/gemm -lcublasGEMM 
LD_FLAGS += -L../../lib/spdkrpc/ -I../../lib/spdkrpc/ -lspdkrpc

TARGET=spdk_gemm
SRC=verify.cu

ifeq ($(debug),1)
	LD_FLAGS += -g -DDEBUG
endif

default: all

all: $(TARGET)

$(TARGET): $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS) -DGATHER

clean:
	rm -rf $(TARGET)

clean_answer:
	rm -rf *.bin
