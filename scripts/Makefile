LD_FLAGS = -L../lib/rdma/ -I../lib/rdma/ -lrdma
PROG=change_config_req

.PHONY: $(PROG)

all: $(PROG)

$(PROG): $(PROG).c
	$(CC) -o $(PROG) $? $(LD_FLAGS)

clean:
	rm $(PROG)