CC = gcc
LD_FLAGS = -O3

TARGET=bfs
SRC=bfs.c

ifeq ($(debug),1)
	LD_FLAGS += -g -DDEBUG
endif

default: all

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -o $@ $< $(LD_FLAGS) 

clean:
	rm -rf $(TARGET)

clean_answer:
	rm -rf *.bin
