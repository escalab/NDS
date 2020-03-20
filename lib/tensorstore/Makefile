CC=gcc
CFLAGS=-O3

TARGET = libtensorstore.so
SRC=tensorstore.c

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

default: $(TARGET)

all: $(TARGET) test

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -fPIC --shared -o $@ $< 

test: test.c $(SRC)
	$(CC) $(CFLAGS) -c $^
	$(CC) $(CFLAGS) -o $@ *.o
	rm -f *.o

clean:
	rm -f $(TARGET) test