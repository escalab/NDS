import numpy
import sys
import json

result_arr = []

for filename in sys.argv[1:-1]:
    with open(filename, 'rb') as f:
        eventname = filename.split('.')[0]

        s = f.read()
        ptr = 0
        numts = numpy.fromstring(s[ptr:ptr+numpy.dtype(numpy.uint64).itemsize], dtype=numpy.uint64)
        ptr += numpy.dtype(numpy.uint64).itemsize
        numts = numts[0]
        print(numts)
        tss = numpy.fromstring(s[ptr:], dtype=numpy.uint64)
        for i in range(numts):
            start_json = {
                'pid': 0,
                'name': eventname,
                'cat': 'foo',
                'pid': 0,
                'tid': 0,
            }
            end_json = {
                'pid': 0,
                'name': eventname,
                'cat': 'foo',
                'pid': 0,
                'tid': 0,
            }
            start_json['ph'] = "B"
            end_json['ph'] = "E"
            start_json['ts'] = int(tss[2 * i])
            end_json['ts'] = int(tss[2 * i + 1])
            result_arr.append(start_json)
            result_arr.append(end_json)

with open(sys.argv[-1], 'w') as outf:
    json.dump(result_arr, outf, indent=4)
