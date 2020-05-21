NVCC = nvcc
LD_FLAGS = -O3 -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcublas_static -lculibos -lcudart_static -lpthread -ldl -lcublasLt_static
# LD_FLAGS += -I../../lib/gemm -L../../lib/gemm -lcublasGEMM

SRC=perftest.cu
TARGET=transport_perftest

ifeq ($(debug),1)
	LD_FLAGS += -g -DDEBUG
endif

default: all

all: $(TARGET)

$(TARGET): $(SRC) 
	$(NVCC) -o $@ $^ $(LD_FLAGS)

clean:
	rm -rf $(TARGET)