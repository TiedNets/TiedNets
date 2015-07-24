__author__ = 'Agostino Sturaro'

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

input_fpath = os.path.normpath('../Simulations/MN_nets/1cc_2ap/rnd_atk/realistic/_stats.tsv')
output_fpath = os.path.normpath('../Simulations/MN_nets/1cc_2ap/rnd_atk/realistic/MN_nets_reason_bars_real_2.pdf')
instance_type = str(0)
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

y_vals = list()
x_labels = list()
y_errs = list()

# create lists of values and errors, iterate over field names to ensure dictionaries are read in the correct order
for key_name in fieldnames:
    if key_name.endswith(val_key_name_suff):
        y_vals.append(desired_line[key_name])
        label_name = key_name[:-len(val_key_name_suff)]   # remove suffix from label names (optional)
        x_labels.append(label_name)
    elif key_name.endswith(err_key_name_suff):
        y_errs.append(desired_line[key_name])

y_vals = np.array(map(float, y_vals))  # convert values to floats
y_errs = np.array(map(float, y_errs))  # convert values to floats

bar_cnt = len(y_vals)  # the number of bars in the graph
width = 0.35  # the width of the bars
bar_pos = np.arange(width, bar_cnt + width)  # the x locations for the groups

fig, ax = plt.subplots()
bars = ax.bar(bar_pos, y_vals, width, color='r', yerr=y_errs)

# dress bars in patterns
patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
for bar, pattern in zip(bars, patterns):
     bar.set_hatch(pattern)

# add some text for labels, title and axes ticks
ax.set_ylabel('No. final dead nodes')  # label of y axis
ax.set_xticks(bar_pos + width / 2)
ax.set_xticklabels(x_labels, rotation=30, ha='right')

plt.ylim(0.0, plt.ylim()[1])  # cap y axis at zero

plt.savefig(output_fpath, bbox_inches='tight')
plt.show()
