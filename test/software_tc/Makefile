NVCC = nvcc
CFLAGS = -O3
LD_FLAGS = -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcublas_static -lculibos -lcudart_static -lpthread -ldl -lcublasLt_static -libverbs
LD_FLAGS += -I../../lib/gemm -L../../lib/gemm -lcublasGEMM 
LD_FLAGS += -L../../lib/timing/ -I../../lib/timing/ -ltiming

TARGET=tc
SRC=verify.cu fifo.c rdma.c setup_rdma.c sock.c

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
