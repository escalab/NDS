NVCC = nvcc
CFLAGS = -O3
LD_FLAGS = -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lpthread -ldl -libverbs
LD_FLAGS += -L../../lib/rdma/ -I../../lib/rdma/ -lrdma

TARGET=close
SRC=close.cu

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
