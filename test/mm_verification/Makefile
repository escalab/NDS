CC=gcc
CFLAGS=-O3

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

all: main

main: main.c
	$(CC) $(CFLAGS) -o $@ $< 

clean:
	rm -rf main