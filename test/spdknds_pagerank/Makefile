CC = gcc
LD_FLAGS = -L../../lib/spdkrpc/ -I../../lib/spdkrpc/ -lspdkrpc
TARGET=pagerank
SRC=pagerank.c

ifeq ($(debug),1)
	LD_FLAGS += -g -DDEBUG
endif

default: all

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -o $@ $< $(LD_FLAGS) 

clean:
	rm -rf $(TARGET)
