CC=gcc
CFLAGS=-O3 -Wall
LD_FLAGS = -I../lib/tensorstore -L../lib/tensorstore -ltensorstore

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

all: datagen_double datagen_long

datagen_double: datagen_double.c
	$(CC) $(CFLAGS) -o $@ $< $(LD_FLAGS)

datagen_long: datagen_long.c
	$(CC) $(CFLAGS) -o $@ $< 

clean:
	rm -f datagen_double datagen_long

clean_data:
	rm -f *.bin