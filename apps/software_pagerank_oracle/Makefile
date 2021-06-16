NVCC = nvcc
CFLAGS = -O3
LD_FLAGS = -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcublas_static -lculibos -lcudart_static -lpthread -ldl -lcublasLt_static
LD_FLAGS += -L../../lib/timing/ -I../../lib/timing/ -ltiming
LD_FLAGS += -L../../lib/rdma/ -I../../lib/rdma/ -lrdma
LD_FLAGS += -L../../lib/fifo/ -I../../lib/fifo/ -lfifo

TARGET=pagerank_oracle
SRC=verify.cu

ifeq ($(debug),1)
	LD_FLAGS += -g -DDEBUG
endif

default: all

all: $(TARGET)

$(TARGET): $(SRC) 
	$(NVCC) $(CFLAGS) -o $@ $^ $(LD_FLAGS)

clean:
	rm -rf $(TARGET)

clean_answer:
	rm -rf *.bin
