LD_FLAGS = -L../lib/rdma/ -I../lib/rdma/ -lrdma
PROG=daemon_response

all: daemonize.c
	$(CC) -o $(PROG) $? $(LD_FLAGS)

clean:
	rm $(PROG)