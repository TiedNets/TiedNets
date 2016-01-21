__author__ = 'Agostino Sturaro'

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import shared_functions as sf

# input_fpath = os.path.normpath('../Simulations/synthetic/synth_rnd_atk_failure_cnt.tsv')
# output_fpath = os.path.normpath('../Simulations/synthetic/synth_rnd_atk_failure_cnt.pdf')
# input_fpath = os.path.normpath('../Simulations/MN_nets/MN_deg_atks_failure_cnt.tsv')
# output_fpath = os.path.normpath('../Simulations/MN_nets/MN_deg_atks_failure_cnt.pdf')
# input_fpath = os.path.normpath('../Simulations/MN_nets/MN_rnd_atk_stats.tsv')
# output_fpath = os.path.normpath('../Simulations/MN_nets/MN_rnd_atk_failure_cnt.pdf')

# each row in the input file is meant to be a point in the graph
input_fpath = os.path.normpath('../Simulations/centrality/1cc_1ap/betw_c/realistic/_stats.tsv')
output_fpath = os.path.normpath('../Simulations/centrality/1cc_1ap/betw_c/realistic/betw_c_graph.pdf')

# the input file structured like a table, we read it all into an array-like structure using Numpy
values = np.genfromtxt(input_fpath, delimiter='\t', skip_header=1, dtype=None)
# print(values)  # debug

# we separate the array by columns, the user should know what column represents what
# remember to correctly specify the number of these columns!
# cells of the first column indicate what instance group (line) the ith row (point) belongs to
group_col = sf.get_unnamed_numpy_col_as_list(values, 0)
x_col = sf.get_unnamed_numpy_col_as_list(values, 1)
y_col = sf.get_unnamed_numpy_col_as_list(values, 2)
error_col = sf.get_unnamed_numpy_col_as_list(values, 3)  # get the column representing the error measure (std dev)
print('\ngroups {}\nX {}\nY {}\nerrors {}'.format(group_col, x_col, y_col, error_col))  # debug

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

groups = set(group_col)  # get the names of the different groups
d = {}  # will contain additional arguments for the plotting call of each line
# idx and val are, respectively, the index and name of a group (line)
for idx, val in enumerate(groups):
    d[val] = {'label': '{}'.format(val), 'marker': col_marks[idx][0], 'color': col_marks[idx][1],
              'markersize': 10, 'linewidth': 1}

# draw a different line for each of the unique values in X
# only works with NumPy arrays
# for val, kwargs in d.items():
#     mask = group_col == val  # create a mask to filter the lists, it's a vector of 0/1 values, 1 if ith cell == val
#     line_x, line_y, line_err = x_col[mask], y_col[mask], error_col[mask]  # use the mask to filter the lists
#     plt.errorbar(line_x, line_y, yerr=line_err, **kwargs)  # this is the call that actually plots the lines

# dictionaries of lists, they will contain the coordinates of the points belonging to each line
x_vals_by_line = {}
y_vals_by_line = {}
errors_by_line = {}
# initialize lists for each line
for line in groups:
    x_vals_by_line[line] = []
    y_vals_by_line[line] = []
    errors_by_line[line] = []

# separate data belonging to each line, look at the cells of the column "group", remember each row looked like this
# group x y err
for idx, line in enumerate(group_col):
    x_vals_by_line[line].append(x_col[idx])
    y_vals_by_line[line].append(y_col[idx])
    errors_by_line[line].append(error_col[idx])

# plot the different lines
for line, kwargs in d.items():
    line_x = x_vals_by_line[line]
    line_y = y_vals_by_line[line]
    line_err = errors_by_line[line]
    plt.errorbar(line_x, line_y, yerr=line_err, **kwargs)  # this is the call that actually plots each line

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
