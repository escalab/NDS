CC=gcc
CFLAGS=-O3

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

all: read_data read_sequential read_block

read_data: read_data.c
	$(CC) $(CFLAGS) -o $@ $< 

read_sequential: read_sequential.c
	$(CC) $(CFLAGS) -o $@ $< 

read_block: read_block.c
	$(CC) $(CFLAGS) -o $@ $< 

clean:
	rm -f read_data read_sequential read_block