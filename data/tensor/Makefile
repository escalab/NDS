CC=gcc
CFLAGS=-O3 -Wall
LD_FLAGS = 
TARGET=tensor_datagen seq2tensor3d

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

SRCS := $(wildcard *.c)
BINS := $(SRCS:%.c=%)

all: ${BINS}

%: %.c
	$(CC) $(CFLAGS) -O3 -o $@ $^ $(LD_FLAGS)

tensor_datagen: tensor_datagen.c
	$(CC) $(CFLAGS) -o $@ $< $(LD_FLAGS)

seq2tensor3d: seq2tensor3d.c
	$(CC) $(CFLAGS) -o $@ $< 

clean:
	rm -f $(TARGET)

clean_data:
	rm -f *.bin