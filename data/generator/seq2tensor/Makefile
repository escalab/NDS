CC=gcc
CFLAGS=-O3 -Wall
LD_FLAGS = -I../lib/tensorstore -L../lib/tensorstore -ltensorstore
TARGET=datagen_long datagen_double datagen_double_seq datagen_seq2tensor datagen_tensor2seq datagen_tensor2colstripe

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

SRCS := $(wildcard *.c)
BINS := $(SRCS:%.c=%)

all: ${BINS}

%: %.c
	$(CC) $(CFLAGS) -O3 -o $@ $^ $(LD_FLAGS)

datagen_double: datagen_double.c
	$(CC) $(CFLAGS) -o $@ $< $(LD_FLAGS)

datagen_long: datagen_long.c
	$(CC) $(CFLAGS) -o $@ $< 

clean:
	rm -f $(TARGET)

clean_data:
	rm -f *.bin