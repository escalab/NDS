CC=gcc
CFLAGS=-O3 -Wall

TARGET = libtensorstore.so
SRC=tensorstore.c
TEST=test_transform test_transpose
ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

default: $(TARGET)

all: $(TARGET) test

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -fPIC --shared -o $@ $< 

# test_transform: test_transform.c $(SRC)
# 	$(CC) $(CFLAGS) -c $^
# 	$(CC) $(CFLAGS) -o $@ *.o
# 	rm -f *.o

test: $(TEST)

%: %.c $(SRC)
	$(CC) $(CFLAGS) -c $^
	$(CC) $(CFLAGS) -o $@ *.o
	rm -f *.o

clean:
	rm -f $(TARGET) $(TEST)