__author__ = 'Agostino Sturaro'

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import shared_functions as sf

input_fpath = os.path.normpath('../Simulations/synthetic/synth_rnd_atk_failure_cnt.tsv')
output_fpath = os.path.normpath('../Simulations/synthetic/synth_rnd_atk_failure_cnt.pdf')
# input_fpath = os.path.normpath('../Simulations/MN_nets/MN_deg_atks_failure_cnt.tsv')
# output_fpath = os.path.normpath('../Simulations/MN_nets/MN_deg_atks_failure_cnt.pdf')
# input_fpath = os.path.normpath('../Simulations/MN_nets/MN_rnd_atk_stats.tsv')
# output_fpath = os.path.normpath('../Simulations/MN_nets/MN_rnd_atk_failure_cnt.pdf')

# read values from file, by column
values = np.genfromtxt(input_fpath, delimiter='\t', skip_header=1, dtype=None)

# each row is a data point
groups = sf.get_unnamed_numpy_col(values, 0)  # the first cell of each line specifies which data group of the point
X = sf.get_unnamed_numpy_col(values, 1)
Y = sf.get_unnamed_numpy_col(values, 2)
errors = sf.get_unnamed_numpy_col(values, 3)
# print('\ngroups ' + str(groups) + '\nX ' + str(X) + '\nY ' + str(Y) + '\nerrors ' + str(errors))

# we want to make a plot like this
# draw a line for all points with x=x0, label it x0
# draw a line for all points with x=x1, label it x1
# draw a line for all points with x=x2, label it x2
# etc.

# read the unique values in X and use them as keys in a dictionary of line properties
# d = {val: {'label': 'deg {}'.format(val), 'linestyle': '--', 'marker': 'o'} for val in set(groups)}

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
markers = ['o', '^', 's', '*', 'x', '+', 'd']
# linestyles = ['--', ':', '--', ':']
col_marks = sf.mix(colors, markers)

d = {}  # additional arguments for the plotting call of each line
# idx and val are, respectively, the number and name of a line (group of data)
for idx, val in enumerate(set(groups)):
    d[val] = {'label': '{}'.format(val), 'marker': col_marks[idx][0], 'color': col_marks[idx][1],
              'markersize': 10, 'linewidth': 1}

# print(d)

# draw a different line for each of the unique values in X
for val, kwargs in d.items():
    mask = groups == val
    y, z, e = X[mask], Y[mask], errors[mask]
    plt.errorbar(y, z, yerr=e, **kwargs)  # this is the call that actually plots the lines

# label the axes of the plot
ax = plt.axes()
ax.set_xlabel('No. initial failures from attacks', fontsize=20)
ax.set_ylabel('No. total failures after cascades', fontsize=20)


# majorFormatter = FormatStrFormatter('%d')

# set the frequency of the ticks on the x axis
x_min_loc = MultipleLocator(10)
ax.xaxis.set_minor_locator(x_min_loc)
x_maj_loc = MultipleLocator(20)
ax.xaxis.set_major_locator(x_maj_loc)

# set the frequency of the ticks on the y axis
y_min_loc = MultipleLocator(50)
ax.yaxis.set_minor_locator(y_min_loc)
y_maj_loc = MultipleLocator(250)
ax.yaxis.set_major_locator(y_maj_loc)
ax.yaxis.grid(True, 'major')  # show grid grid (dashed lines crossing the graph where major ticks are)

plt.tick_params(axis='both', which='major', labelsize=18)
plt.tick_params(axis='both', which='minor', labelsize=8)

# get the labels of all the lines in the graph
handles, labels = ax.get_legend_handles_labels()

# sort labels to be used for creating the legend
handles, labels = zip(*sorted(zip(handles, labels), key=lambda x: x[1]))

# create a legend growing it from the middle and put it on the right side of the graph
# lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(0.99, 0.5), fontsize=16)

# lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.14), fontsize=10)
# lgd = ax.legend(handles, labels, loc='center', bbox_to_anchor=(0.05, 0.6), fontsize=12)
# lgd = ax.legend(handles, labels, loc='center', bbox_to_anchor=(0.35, 0.70), fontsize=12)
# lgd = ax.legend(handles, labels, loc=4, fontsize=12)
lgd = ax.legend(handles, labels, bbox_to_anchor=(-0.11, 1., 1.11, 0.), loc=3,
           ncol=4, mode="expand", borderaxespad=0., fontsize=16)

plt.ylim(0.0, 2100.0)  # cap y axis at zero

# save the figure so that the legend fits inside it
plt.savefig(output_fpath, bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()
