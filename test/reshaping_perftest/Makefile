CC=gcc
CFLAGS=-O3 -Wall
LDFLAG=-L../../lib/tensorstore/ -I../../lib/tensorstore/ -ltensorstore

TARGET=test_transform test_transform_col_stripe test_transform_row_stripe

ifeq ($(debug),1)
	CFLAGS += -g -DDEBUG
endif

all: $(TARGET)

%: %.c $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAG)

clean:
	rm -f $(TARGET)