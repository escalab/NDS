CC=gcc
CFLAGS=-O3 -Wall

TARGET=libfifo.a
SRC=fifo.c

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

default: $(TARGET)

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -c $<
	ar rcs $@ *.o
	rm *.o

clean:
	rm -f $(TARGET)