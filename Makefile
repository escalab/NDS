CC=gcc
CFLAGS=-O3

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

all: datagen read_data read_sequential read_block

datagen: datagen.c
	$(CC) $(CFLAGS) -o $@ $< 

read_data: read_data.c
	$(CC) $(CFLAGS) -o $@ $< 

read_sequential: read_sequential.c
	$(CC) $(CFLAGS) -o $@ $< 

read_block: read_block.c
	$(CC) $(CFLAGS) -o $@ $< 

clean:
	rm -f datagen read_data read_sequential read_block