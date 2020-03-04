CC=gcc
CFLAGS=-O3

all: datagen read_data

datagen: datagen.c
	$(CC) $(CFLAGS) -o $@ $< 

read_data: read_data.c
	$(CC) $(CFLAGS) -o $@ $< 

clean:
	rm -f datagen read_data