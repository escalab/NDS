NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include  -lcublas_static -lculibos -lcudart_static -lpthread -ldl -lcublasLt_static
LD_FLAGS    = -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcublas_static -lculibos -lcudart_static -lpthread -ldl -lcublasLt_static
TARGET      = cublas_gemm_test
OBJ 		= main.o 
SRC			= main.cu

ifeq ($(debug),1)
	LD_FLAGS += -g -DDEBUG
endif

default: $(TARGET)

all: $(OBJ) $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $@ $< $(LD_FLAGS)

%.o: %.cu
	$(NVCC) -c -o $@ $< $(NVCC_FLAGS)

clean:
	rm -rf $(OBJ) $(TARGET)
