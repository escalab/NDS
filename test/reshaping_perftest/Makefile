CC=gcc
CFLAGS=-O3 -Wall
LDFLAG=-L../../lib/tensorstore/ -I../../lib/tensorstore/ -ltensorstore

TARGET=test_transform

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

all: $(TARGET)

%: %.c $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAG)

clean:
	rm -f $(TARGET)