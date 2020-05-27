CC=gcc
CFLAGS=-O3 -Wall 
LDFLAGS=-L/usr/local/lib -lcjson

TARGET=spdkrpc.so
SRC=sock.c mmap_reader.c json_parser.c api.c

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

default: $(TARGET)

all: $(TARGET) test

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -fPIC --shared -o $@ $^ $(LDFLAGS)

test: test.c $(SRC)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) test