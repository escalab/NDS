CC=gcc
CFLAGS=-O3 -Wall
LDFLAG=-L../../lib/tensorstore/ -I../../lib/tensorstore/ -ltensorstore
LDFLAG+=-L../../lib/spdkrpc/ -I../../lib/spdkrpc/ -lspdkrpc

TARGET=test_transpose test_transpose_spdk

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

all: $(TARGET)

%: %.c $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAG)

clean:
	rm -f $(TARGET)