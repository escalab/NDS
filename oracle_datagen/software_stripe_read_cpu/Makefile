NVCC = gcc
CFLAGS = -O3
LD_FLAGS = -lpthread -ldl
# LD_FLAGS += -I../../lib/gemm -L../../lib/gemm -lcublasGEMM 
# LD_FLAGS = -lcuda -lcublas -lcudart  -lpthread -ldl
LD_FLAGS += -L../../lib/timing/ -I../../lib/timing/ -ltiming
LD_FLAGS += -L../../lib/rdma/ -I../../lib/rdma/ -lrdma
LD_FLAGS += -L../../lib/fifo/ -I../../lib/fifo/ -lfifo

TARGET=stripe_read
SRC=verify.c

ifeq ($(debug),1)
	LD_FLAGS += -g -DDEBUG
endif

ifdef sub_m
	CFLAGS += -DSUB_M=$(sub_m)UL
endif

default: all

all: $(TARGET)

$(TARGET): $(SRC) 
	$(NVCC) $(CFLAGS) -o $@ $^ $(LD_FLAGS)

clean:
	rm -rf $(TARGET)

clean_answer:
	rm -rf *.bin
