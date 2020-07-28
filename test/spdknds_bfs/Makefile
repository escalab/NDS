CC = nvcc
LD_FLAGS = -L../../lib/spdkrpc/ -I../../lib/spdkrpc/ -lspdkrpc
TARGET=bfs
SRC=bfs.cu

ifeq ($(debug),1)
	LD_FLAGS += -g -DDEBUG
endif

default: all

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -o $@ $< $(LD_FLAGS) 

clean:
	rm -rf $(TARGET)
