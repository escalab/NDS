CC=gcc
CFLAGS=-O3 -Wall
LDFLAG=-L../../lib/tensorstore/ -I../../lib/tensorstore/ -ltensorstore

TARGET=test_transpose

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

all: test_transpose

%: %.c $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAG)

clean:
	rm -f $(TARGET)