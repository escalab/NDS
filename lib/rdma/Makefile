CC=gcc
CFLAGS=-O3 -Wall 
LDFLAGS=-L/usr/local/lib

TARGET=librdma.a
SRC=rdma.c setup_rdma.c sock.c

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

default: $(TARGET)

all: $(TARGET) 

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -c $^ $(LDFLAGS)
	ar rcs $@ *.o
	rm *.o

clean:
	rm -f $(TARGET)