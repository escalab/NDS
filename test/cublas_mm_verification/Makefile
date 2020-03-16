NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include  -lcublas_static -lculibos -lcudart_static -lpthread -ldl -lcublasLt_static
LD_FLAGS    = -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcublas_static -lculibos -lcudart_static -lpthread -ldl -lcublasLt_static
TARGET      = cublas_gemm_test
OBJ 		= main.o support.o common.o cublas_gemm_kernel.o
SRC			= main.cu support.cu common.cu cublas_gemm_kernel.cu
VERIFY_TARGET = cublas_gemm_verify
VERIFY_SRC	= verify.cu support.cu common.cu cublas_gemm_kernel.cu


default: $(TARGET)

all: $(OBJ) $(TARGET) $(VERIFY_TARGET)

$(VERIFY_TARGET): $(VERIFY_SRC)
	$(NVCC) -o $@ $< $(LD_FLAGS)

$(TARGET): $(SRC)
	$(NVCC) -o $(TARGET) $< $(LD_FLAGS)

%.o: %.cu
	$(NVCC) -c -o $@ $< $(NVCC_FLAGS)

clean:
	rm -rf $(OBJ) $(TARGET) $(VERIFY_TARGET)
