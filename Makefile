CC=gcc
CFLAGS=-O3

all: datagen read_data read_sequential

datagen: datagen.c
	$(CC) $(CFLAGS) -o $@ $< 

read_data: read_data.c
	$(CC) $(CFLAGS) -o $@ $< 

read_sequential: read_sequential.c
	$(CC) $(CFLAGS) -o $@ $< 

clean:
	rm -f datagen read_data read_sequential