CC=g++
CFLAGS=-O3

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

all: main

main: main.cpp
	$(CC) $(CFLAGS) -o $@ $< 

clean:
	rm -rf main