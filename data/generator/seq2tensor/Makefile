CC=gcc
CFLAGS=-O3 -Wall
TARGET= datagen_seq2tensor

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

SRCS := $(wildcard *.c)
BINS := $(SRCS:%.c=%)

all: ${BINS}

%: %.c
	$(CC) $(CFLAGS) -O3 -o $@ $^

clean:
	rm -f $(TARGET)

clean_data:
	rm -f *.bin