NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -lcublas_static -lculibos -lcudart_static -lpthread -ldl -lcublasLt_static
# LD_FLAGS    = -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcublas_static -lculibos -lcudart_static -lpthread -ldl -lcublasLt_static

TARGET=libcublasGEMM.so
SRC=cublasGEMM.cu

ifeq ($(debug),1)
	LD_FLAGS += -g -DDEBUG
endif

default: all

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) --compiler-options '-fPIC' -o $@ --shared $< $(NVCC_FLAGS)

clean:
	rm -rf $(TARGET)
