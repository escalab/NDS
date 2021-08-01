import numpy
import re
import pandas
from scipy import stats

pandas.options.display.float_format = "{:.3f}".format

configuration_list = ['baseline', 'software_nds', 'hardware_nds']
app_list = ['block-GEMM', 'k-means', 'k-nn', 'bfs', 'bellman-ford', 'pagerank', 'conv2d', 'hotspot', 'ttv', 'tc']

num_configurations = len(configuration_list)
num_applications = len(app_list)

arr = numpy.zeros((num_configurations, num_applications), dtype=numpy.float32)

for i, config in enumerate(configuration_list):
    with open('{}.txt'.format(config), 'r') as f:
        count = 0
        for line in f:
            m = re.search('End-to-end duration: ([-+]?[0-9]*\.[0-9]+|[0-9]+) ms', line)
            if m:
                # from msec to sec
                arr[i, count] = float(m.group(1)) / 1000.0
                count += 1


print('End-to-end Duration Table (secs):')
time_df = pandas.DataFrame(data=arr, index=configuration_list, columns=app_list)
print(time_df.transpose())
print()


speed_arr = numpy.zeros((num_configurations-1, num_applications), dtype=numpy.float32)
speed_arr[0] = arr[0] / arr[1]
speed_arr[1] = arr[0] / arr[2]
speedup_df = pandas.DataFrame(data=speed_arr, index=configuration_list[1:], columns=app_list)

print('Speedup Table (Software SEQ as baseline):')
summary_df = speedup_df.copy()
summary_df[''] = ''
summary_df['average'] = speedup_df.mean(numeric_only=True, axis=1)
summary_df['geomean'] = stats.gmean(speedup_df, axis=1)
print(summary_df.transpose())
