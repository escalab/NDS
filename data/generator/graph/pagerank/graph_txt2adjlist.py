import sys
import numpy

if len(sys.argv) < 2:
    print("usage: {} <graph txt>".format(sys.argv[0]))
    exit(1)

outfile_name = "{}.adjlist".format(sys.argv[1].split('.')[0])

with open(sys.argv[1], 'r') as fin:
    with open(outfile_name, 'w') as fout:
        first_line = fin.readline().split()
        num_vertex = int(first_line[0])
        num_edge = int(first_line[1]) * 2 # undirected graph
        edges = numpy.zeros(num_vertex, dtype=numpy.int64)
        count = 0
        for vid, line in enumerate(fin):
            outedge = line.split()
            count += len(outedge)
            fout.write("{} {}".format(vid, len(outedge)))
            for edge in outedge:
                fout.write(" {}".format(int(edge)-1))
            fout.write("\n")
        print(count == num_edge)
