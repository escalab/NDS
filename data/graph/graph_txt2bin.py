import sys
import numpy

if len(sys.argv) < 2:
    print("usage: {} <graph txt>".format(sys.argv[0]))
    exit(1)

outfile_name = "{}.bin".format(sys.argv[1].split('.')[0])

with open(sys.argv[1], 'r') as fin:
    with open(outfile_name, 'wb') as fout:
        first_line = fin.readline().split()
        num_vertex = int(first_line[0])
        num_edge = int(first_line[1]) * 2 # undirected graph
        edges = numpy.zeros(num_vertex, dtype=numpy.int64)
        count = 0
        for line in fin:
            for idx in line.split():
                edges[int(idx)-1] = 1
                count += 1
            fout.write(edges.tobytes())
            edges.fill(0)
        print(count == num_edge)
