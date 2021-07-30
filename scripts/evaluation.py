import numpy
import re
import pandas
from scipy import stats

pandas.options.display.float_format = "{:.3f}".format

configuration_list = ['software_seq', 'software_nds', 'hardware_nds']
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

df = pandas.DataFrame(data=arr, index=configuration_list, columns=app_list)
summary_df = df.copy()
summary_df[''] = ''
summary_df['average'] = df.mean(numeric_only=True, axis=1)
summary_df['geomean'] = stats.gmean(df, axis=1)
print('result table (secs):')
print(summary_df.transpose())

print()
print('software NDS speedup: {:.3f}x'.format(sum(arr[0]) / sum(arr[1])))
print('hardware NDS speedup: {:.3f}x'.format(sum(arr[0]) / sum(arr[2])))
print('hardware NDS speedup compared to software NDS: {:.3f}x'.format(sum(arr[1]) / sum(arr[2])))