import sys
import numpy

if len(sys.argv) < 2:
    print("usage: {} <graph txt>".format(sys.argv[0]))
    exit(1)

outfile_name = "{}".format(sys.argv[1].split('.')[0])

with open(sys.argv[1], 'r') as fin:
    first_line = fin.readline().split()
    num_vertex = int(first_line[0])
    num_edge = int(first_line[1]) * 2 # undirected graph
    col = numpy.zeros(num_edge, dtype=numpy.int64)
    row = numpy.zeros(num_vertex+1, dtype=numpy.int64)
    count = 0
    edge_count = 0
    for vid, line in enumerate(fin):
        outedge = line.split()
        count += len(outedge)
        row[vid+1] = count
        for edge in outedge:
            col[edge_count] = int(edge) - 1
            edge_count += 1
    print(edge_count == num_edge)
    print(count == num_edge)

with open("{}.col".format(outfile_name), 'w') as fout:
    for c in col:
        fout.write("{}\n".format(c))

with open("{}.row".format(outfile_name), 'w') as fout:
    for r in row:
        fout.write("{}\n".format(r))