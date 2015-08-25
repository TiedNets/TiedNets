__author__ = 'Agostino Sturaro'

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

input_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/rnd_atk/realistic/_stats_ext.tsv')
output_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/rnd_atk/realistic/MN_rnd_atk_reasons_real_1_ap.pdf')
instance_type = str('Realistic 1')
indep_var_val = str(40)
val_key_name_suff = '_avg'
err_key_name_suff = '_std'

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

bar_height = list()
bar_names = list()
bar_errs = list()

# create lists of values and errors, iterate over field names to ensure dictionaries are read in the correct order
for key_name in fieldnames:
    if key_name.endswith(val_key_name_suff):
        bar_height.append(desired_line[key_name])
        bar_name = key_name[:-len(val_key_name_suff)]   # remove suffix from label names (optional)
        bar_names.append(bar_name)
    elif key_name.endswith(err_key_name_suff):
        bar_errs.append(desired_line[key_name])

bar_height = np.array(map(float, bar_height))  # convert values to floats
bar_errs = np.array(map(float, bar_errs))  # convert values to floats

bar_cnt = len(bar_height)  # the number of bars in the graph
bar_width = 0.35  # the width of the bars
bar_pos = np.arange(bar_width, bar_cnt + bar_width)  # the x locations for the groups

patterns = ('*', '+', 'x', '\\', '-', 'o', 'O', '.')
fig, ax = plt.subplots()
# draw bars
for i in range(len(bar_height)):
    bars = ax.bar(bar_pos[i], bar_height[i], bar_width, color='w', label=bar_names[i], hatch=patterns[i])

# add some text for labels, title and axes ticks
ax.set_ylabel('No. final dead nodes')  # label of y axis
ax.set_xticks(bar_pos + bar_width / 2)
# turn off minor and major (both) tick marks on the x axis
plt.tick_params(axis='x', which='both', labelbottom='off')
# ax.set_xticklabels(bar_names, rotation=30, ha='right')  # draw a label below each bar (optional)

# get the labels of all the bars in the graph
handles, labels = ax.get_legend_handles_labels()

# lgd = ax.legend(handles, labels, bbox_to_anchor=(-0.11, 1., 1.11, 0.), loc=3,
#            ncol=4, mode="expand", borderaxespad=0., fontsize=16)
lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(0.99, 0.5), fontsize=16)

plt.ylim(0.0, plt.ylim()[1])  # cap y axis at zero

plt.savefig(output_fpath, bbox_inches='tight')
plt.show()
