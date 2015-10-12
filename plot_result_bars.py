__author__ = 'Agostino Sturaro'

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# BEGIN user defined variables
input_fpath = os.path.normpath('../Simulations/MN_nets/MN_rnd_atk_reasons.tsv')
output_fpath = os.path.normpath('../Simulations/MN_nets/MN_rnd_atk_reasons.pdf')

# we need a single table row as input, identifying a set of experiments on the same instance type, done with the same
# value of the independent variable
instance_type = 'HINT'
indep_var_val = str(40)
val_key_name_suff = '_avg'
err_key_name_suff = '_std'
# END user defined variables

# adjustments to ensure relative paths will work
if not os.path.isabs(input_fpath) or not os.path.isabs(output_fpath):
    this_dir = os.path.normpath(os.path.dirname(__file__))
    os.chdir(this_dir)

# ensure that the specified input file exists
if os.path.isfile(input_fpath) is not True:
    raise ValueError('Invalid value for parameter input_fpath')

# read values from file, by column
with open(input_fpath) as input_file:
    values = csv.DictReader(input_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)

    fieldnames = list(values.fieldnames)
    fieldnames.remove('Instance_type')
    fieldnames.remove('Indep_var_val')

    for line in values:
        if line['Instance_type'] == instance_type:
            if line['Indep_var_val'] == indep_var_val:
                desired_line = line
                desired_line.pop('Instance_type')
                desired_line.pop('Indep_var_val')
                break

bar_heights = list()
bar_names = list()
bar_errs = list()

# create lists of values and errors, iterate over field names to ensure dictionaries are read in the correct order
for key_name in fieldnames:
    if key_name.endswith(val_key_name_suff):
        bar_heights.append(float(desired_line[key_name]))  # cast to float
        bar_name = key_name[:-len(val_key_name_suff)]   # remove suffix from label names (optional)
        bar_names.append(bar_name)
    elif key_name.endswith(err_key_name_suff):
        bar_errs.append(float(desired_line[key_name]))  # cast to float

# if we want a bar with the independent variable (optional)
bar_heights.insert(0, float(indep_var_val))
bar_names.insert(0, 'Initial attack on A')
bar_errs.insert(0, 0.0)

# if we want to express values as percentage of a whole (optional)
whole_value_pos = bar_names.index('Total_dead')
whole_value = bar_heights[whole_value_pos]
del bar_heights[whole_value_pos]
del bar_names[whole_value_pos]
del bar_errs[whole_value_pos]
for i in range(len(bar_heights)):
    bar_heights[i] /= whole_value
    bar_errs[i] /= whole_value

# use a space to separate words (optional)
for i in range(len(bar_names)):
    bar_names[i] = bar_names[i].replace('_', ' ')

bar_cnt = len(bar_heights)  # the number of bars in the graph
bar_width = 0.35  # the width of the bars
bar_pos = np.arange(bar_width, bar_cnt + bar_width)  # the x locations for the groups

patterns = ('*', '+', 'x', '\\', '-', 'o', 'O', '.')
fig, ax = plt.subplots()
# draw bars
for i in range(len(bar_heights)):
    bars = ax.bar(bar_pos[i], bar_heights[i], bar_width, color='w', label=bar_names[i], hatch=patterns[i])

# add some text for labels, title and axes ticks
ax.set_ylabel('Percentage of final dead nodes')  # label of y axis
ax.set_xticks(bar_pos + bar_width / 2)
# turn off minor and major (both) tick marks on the x axis
plt.tick_params(axis='x', which='both', labelbottom='off')
# ax.set_xticklabels(bar_names, rotation=30, ha='right')  # draw a label below each bar (optional)

# get the labels of all the bars in the graph
handles, labels = ax.get_legend_handles_labels()

# lgd = ax.legend(handles, labels, bbox_to_anchor=(-0.11, 1.02, 1.11, 0.), loc=3,
#                 ncol=3, mode="expand", borderaxespad=0., fontsize=16)
# lgd = ax.legend(handles, labels, bbox_to_anchor=(-0.11, -0.02, 1.11, 0.), loc=1,
#                 ncol=3, mode="expand", borderaxespad=0., fontsize=16)
lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(0.99, 0.5), fontsize=16)

plt.ylim(0.0, plt.ylim()[1])  # cap y axis at zero

plt.savefig(output_fpath, bbox_inches='tight')
plt.show()
