__author__ = 'Agostino Sturaro'

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import shared_functions as sf

input_fpath = os.path.normpath('../Simulations/synthetic_nets/rnd_atk_stats.tsv')
output_fpath = os.path.normpath('../Simulations/synthetic_nets/synth_rnd_atk_failure_cnt.pdf')
# input_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/deg_atks/deg_atks_stats.tsv')
# output_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/deg_atks/deg_atks_stats.pdf')

# read values from file, by column
values = np.genfromtxt(input_fpath, delimiter='\t', skip_header=1, dtype=None)

groups = sf.get_unnamed_numpy_col(values, 0)
X = sf.get_unnamed_numpy_col(values, 1)
Y = sf.get_unnamed_numpy_col(values, 10)
errors = sf.get_unnamed_numpy_col(values, 11)
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

d = {}
for idx, val in enumerate(set(groups)):
    d[val] = {'label': '{}'.format(val),
              'marker': col_marks[idx][0], 'color': col_marks[idx][1]}

# print(d)

# draw a different line for each of the unique values in X
for val, kwargs in d.items():
    mask = groups == val
    y, z, e = X[mask], Y[mask], errors[mask]
    plt.errorbar(y, z, yerr=e, **kwargs)

# label the axes of the plot
ax = plt.axes()
ax.set_xlabel('No. initial failures from attacks')
ax.set_ylabel('No. total failures after cascades')

# for the minor ticks, use no labels; default NullFormatter
# x_min_loc = MultipleLocator(1)
# ax.xaxis.set_minor_locator(x_min_loc)
#
y_min_loc = MultipleLocator(5)
ax.yaxis.set_minor_locator(y_min_loc)
ax.yaxis.grid(True)

# get the labels of all the lines in the graph
handles, labels = ax.get_legend_handles_labels()

# sort labels to be used for creating the legend
handles, labels = zip(*sorted(zip(handles, labels), key=lambda x: x[1]))

# create a legend growing it from the middle and put it on the right side of the graph
lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.14), fontsize=10)

plt.ylim(0.0, 2100.0)  # cap y axis at zero

# save the figure so that the legend fits inside it
plt.savefig(output_fpath, bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()
