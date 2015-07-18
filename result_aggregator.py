__author__ = 'Agostino Sturaro'

import os
import csv
import logging
import numpy as np
import shared_functions as sf

this_dir = os.path.normpath(os.path.dirname(__file__))
os.chdir(this_dir)
sf.setup_logging('logging_base_conf.json')
logger = logging.getLogger(__name__)

index_fpath = os.path.normpath('../Simulations/exp_1000n_many/rnd_atk/1_cc/realistic/_index.tsv')
aggregate_fpath = os.path.normpath('../Simulations/exp_1000n_many/rnd_atk/1_cc/realistic/_stats.tsv')

with open(index_fpath, 'r') as index_file, open(aggregate_fpath, 'wb') as aggregate_file:
    first_it = True
    index = csv.DictReader(index_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)

    for line in index:
        stats_fpath = line['Results_file']
        stats = np.genfromtxt(stats_fpath, delimiter='\t', names=True, dtype=None)
        stat_names = stats.dtype.names
        avgs = dict()
        std_devs = dict()

        # assuming all stats files have the same header
        if first_it is True:
            aggregate_header = ['Instance_type', 'Indep_var_val']
            avg_names = [x + '_avg' for x in stat_names]
            std_names = [x + '_std' for x in stat_names]
            sorted_names = sorted(avg_names + std_names)
            aggregate_header.extend(sorted_names)
            aggregate = csv.DictWriter(aggregate_file, aggregate_header, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            aggregate.writeheader()
            first_it = False

        for name in stat_names:
            values = stats[name]  # access by column
            avgs[name + '_avg'] = np.average(values)
            std_devs[name + '_std'] = np.std(values)

        row_cells = dict()
        row_cells['Instance_type'] = line['Instance_type']
        row_cells['Indep_var_val'] = line['Indep_var_val']
        row_cells.update(avgs)
        row_cells.update(std_devs)

        aggregate.writerow(row_cells)
