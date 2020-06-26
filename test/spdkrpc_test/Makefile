CC=gcc
CFLAGS=-O3 -Wall
LDFLAG=-L../../lib/tensorstore/ -I../../lib/tensorstore/ -ltensorstore
LDFLAG+=-L../../lib/spdkrpc/ -I../../lib/spdkrpc/ -lspdkrpc

TARGET=test_get_matrix test_get_gather_matrix
SRC=test.c

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

all: $(TARGET)

test_get_matrix: $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAG)

test_get_gather_matrix: $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAG) -DGATHER

clean:
	rm -f $(TARGET)