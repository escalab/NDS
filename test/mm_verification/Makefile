CC=g++
CFLAGS=-O3

SRC=main.cpp
TARGET=main

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< 

clean:
	rm -rf $(TARGET)

clean_answer:
	rm -rf *.bin