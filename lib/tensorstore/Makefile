CC=gcc
CFLAGS=-O3 -Wall

TARGET = libtensorstore.a
SRC=tensorstore.c
TEST=test_transform
ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

default: $(TARGET)

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -c $^ $(LDFLAGS)
	ar rcs $@ *.o
	rm *.o

$(TEST): $(TEST).c

clean:
	rm -f $(TARGET) $(TEST)