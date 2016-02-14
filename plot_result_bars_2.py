import math
import random
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Agostino'

# draw a bar chart that looks like a "histogram" with error bars
# note that a histogram shows how many points fall on a piece of the x axis
# instead, this averages y values of points that fall on a piece of the x axis

bin_width = .2
bottom = 5.
top = 8.

data_x = [random.uniform(bottom, top) for i in range(10)]
data_y = [random.uniform(bottom, top) for i in range(10)]
n_bins = int(math.ceil(((top - bottom) / bin_width)))

# separate data into bins
binned_data = [[] for i in range(n_bins)]
for i in range(0, len(data_x)):
    pt_x = data_x[i]
    if pt_x < bottom or pt_x >= top:
        print 'out of range'
        continue
    bin_id = int(math.floor(n_bins * (pt_x - bottom) / (top - bottom)))
    pt_y = data_y[i]
    binned_data[bin_id].append(pt_y)

print(binned_data)

# calculate the average and the standard deviation of data in each bin
bin_avgs = []
bin_stdevs = []
for bin in binned_data:
    if len(bin) > 0:
        avg = sum(bin)/len(bin)
        stdev = np.std(bin)
    else:
        avg = 0
        stdev = 0
    bin_avgs.append(avg)
    bin_stdevs.append(stdev)

# position bins and plot them
bin_edges = [bottom + j * bin_width for j in range(len(binned_data))]
plt.bar(bin_edges, bin_avgs, width=bin_width, yerr=bin_stdevs)
plt.show()