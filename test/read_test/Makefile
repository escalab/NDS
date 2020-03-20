CC=gcc
CFLAGS=-O3

TARGET = read_data read_sequential read_block
ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

all: $(TARGET)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< 

clean:
	rm -f $(TARGET)