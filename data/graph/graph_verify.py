import sys
import numpy

if len(sys.argv) < 2:
    print("usage: {} <graph txt> <graph bin>".format(sys.argv[0]))
    exit(1)


with open(sys.argv[1], 'r') as ftxt:
    with open(sys.argv[2], 'rb') as fbin:
        first_line = ftxt.readline().split()
        num_vertex = int(first_line[0])
        num_edge = int(first_line[1]) * 2 # undirected graph
        edges_from_txt = numpy.zeros(num_vertex, dtype=numpy.int64)
        for line in ftxt:
            for idx in line.split():
                edges_from_txt[int(idx)-1] = 1
            edges_from_bin = numpy.frombuffer(fbin.read(num_vertex * 8), dtype=numpy.int64)
            if not numpy.array_equal(edges_from_bin, edges_from_txt):
                print("not equal")
                print(edges_from_bin)
                print(edges_from_txt)
                exit(1)
            edges_from_txt = numpy.zeros(num_vertex, dtype=numpy.int64)

    print("passed")