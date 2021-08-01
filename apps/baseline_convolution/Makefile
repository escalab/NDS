NVCC = nvcc
CFLAGS = -O3
LD_FLAGS = -I/usr/local/cuda/include -L/usr/local/cuda/lib64/  -lculibos -lcudart_static -lpthread -ldl -libverbs
LD_FLAGS += -I../../lib/gemm -L../../lib/gemm -lcublasGEMM 
LD_FLAGS += -L../../lib/timing/ -I../../lib/timing/ -ltiming
LD_FLAGS += -L../../lib/fifo/ -I../../lib/fifo/ -lfifo
LD_FLAGS += -L../../lib/rdma/ -I../../lib/rdma/ -lrdma

TARGET=convolution
SRC=verify.cu convolutionSeparable.cu

ifeq ($(debug),1)
	LD_FLAGS += -g -DDEBUG
endif

default: all

all: $(TARGET)

$(TARGET): $(SRC) 
	$(NVCC) $(CFLAGS) -o $@ $^ $(LD_FLAGS) -DGATHER

clean:
	rm -rf $(TARGET)

clean_answer:
	rm -rf *.bin
