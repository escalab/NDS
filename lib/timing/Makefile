CC=gcc
CFLAGS=-O3 -Wall

TARGET = libtiming.so
SRC=timing.c

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

default: $(TARGET)

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -fPIC --shared -o $@ $< 

clean:
	rm -f $(TARGET)