import os
import csv
import logging
import numpy as np
import shared_functions as sf

__author__ = 'Agostino Sturaro'

this_dir = os.path.normpath(os.path.dirname(__file__))
os.chdir(this_dir)
sf.setup_logging('logging_base_conf.json')
logger = logging.getLogger(__name__)


def run(index_fpath, aggregate_fpath, instance_type_names=None, cols_to_ignore=None):
    with open(index_fpath, 'r') as index_file, open(aggregate_fpath, 'wb') as aggregate_file:
        first_it = True
        # the index file contains the paths of the files with the actual statistics
        # open the index file as a list of dictionaries (kinda, each file line can be accessed as a dictionary)
        index = csv.DictReader(index_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)

        for line in index:
            stats_fpath = line['Results_file']
            # using NumPy, open a statistics file as a dictionary of vectors (kinda, it's indexed by column)
            # NumPy guesses the type of the cells
            stats = np.genfromtxt(stats_fpath, delimiter='\t', names=True, dtype=None)
            stat_names = stats.dtype.names
            avgs = dict()
            std_devs = dict()

            # assuming all stats files have the same header
            if first_it is True:
                aggregate_header = ['Instance_type', 'Indep_var_val']
                for col_name in stat_names:
                    if cols_to_ignore is not None and col_name in cols_to_ignore:
                        continue
                    avg_col_name = col_name + '_avg'
                    if avg_col_name in stat_names:
                        raise ValueError('Name intended for the average of a column is not free to use')
                    aggregate_header.append(avg_col_name)
                    std_col_name = col_name + '_std'
                    if std_col_name in stat_names:
                        raise ValueError('Name intended for the standard deviation of a column is not free to use')
                    aggregate_header.append(std_col_name)
                # avg_names = [x + '_avg' for x in stat_names]
                # std_names = [x + '_std' for x in stat_names]
                # sorted_names = sorted(avg_names + std_names)
                # aggregate_header.extend(sorted_names)
                aggregate = csv.DictWriter(aggregate_file, aggregate_header, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
                aggregate.writeheader()
                first_it = False

            for col_name in stat_names:
                if cols_to_ignore is not None and col_name in cols_to_ignore:
                    continue
                values = stats[col_name]  # access by column
                avgs[col_name + '_avg'] = np.average(values)
                std_devs[col_name + '_std'] = np.std(values)

            row_cells = dict()
            if instance_type_names is None:
                row_cells['Instance_type'] = line['Instance_type']
            else:
                row_cells['Instance_type'] = instance_type_names[line['Instance_type']]
            row_cells['Indep_var_val'] = line['Indep_var_val']
            row_cells.update(avgs)
            row_cells.update(std_devs)

            aggregate.writerow(row_cells)


# instance_type_names = {'0': 'Realistic'}
# index_fpath = os.path.normpath('../Simulations/synthetic/1cc_1ap/rnd_atk/realistic/_index.tsv')
# aggregate_fpath = os.path.normpath('../Simulations/synthetic/1cc_1ap/rnd_atk/realistic/_stats.tsv')
# instance_type_names = {'0': r'SC $\Delta$=20'}
# index_fpath = os.path.normpath('../Simulations/synthetic/1cc_1ap/rnd_atk/sc_th_21/_index.tsv')
# aggregate_fpath = os.path.normpath('../Simulations/synthetic/1cc_1ap/rnd_atk/sc_th_21/_stats.tsv')
# instance_type_names = {'0': r'SC $\Delta$=200'}
# index_fpath = os.path.normpath('../Simulations/synthetic/1cc_1ap/rnd_atk/sc_th_201/_index.tsv')
# aggregate_fpath = os.path.normpath('../Simulations/synthetic/1cc_1ap/rnd_atk/sc_th_201/_stats.tsv')
# instance_type_names = {'0': 'Uniform'}
# index_fpath = os.path.normpath('../Simulations/synthetic/1cc_1ap/rnd_atk/uniform/_index.tsv')
# aggregate_fpath = os.path.normpath('../Simulations/synthetic/1cc_1ap/rnd_atk/uniform/_stats.tsv')

instance_type_names = {'0': 'Betweenness c.'}
index_fpath = os.path.normpath('../Simulations/centrality/1cc_1ap/betw_c/realistic/_index.tsv')
aggregate_fpath = os.path.normpath('../Simulations/centrality/1cc_1ap/betw_c/realistic/_stats.tsv')

# instance_type_names = {'0': 'Inter-degree distribution'}
# index_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/deg_atks/inter_subst_atk/realistic/_index.tsv')
# aggregate_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/deg_atks/inter_subst_atk/realistic/_stats.tsv')
# instance_type_names = {'0': 'Intra-degree distribution'}
# index_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/deg_atks/intra_subst_atk/realistic/_index.tsv')
# aggregate_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/deg_atks/intra_subst_atk/realistic/_stats.tsv')
# instance_type_names = {'0': 'Intra-degree transmission'}
# index_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/deg_atks/intra_tran_atk/realistic/_index.tsv')
# aggregate_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/deg_atks/intra_tran_atk/realistic/_stats.tsv')
# instance_type_names = {'0': 'Intra-degree generator'}
# index_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/deg_atks/intra_gen_atk/realistic/_index.tsv')
# aggregate_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/deg_atks/intra_gen_atk/realistic/_stats.tsv')

# instance_type_names = {'0': 'Realistic 2'}
# index_fpath = os.path.normpath('../Simulations/MN_nets/1cc_2ap/rnd_atk/realistic/_index.tsv')
# aggregate_fpath = os.path.normpath('../Simulations/MN_nets/1cc_2ap/rnd_atk/realistic/_stats.tsv')
# instance_type_names = {'0': 'Realistic 1'}
# index_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/rnd_atk/realistic/_index.tsv')
# aggregate_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/rnd_atk/realistic/_stats.tsv')
# instance_type_names = {'0': 'SC th=21'}
# index_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/rnd_atk/sc_th_21/_index.tsv')
# aggregate_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/rnd_atk/sc_th_21/_stats.tsv')
# instance_type_names = {'0': 'SC th=210'}
# index_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/rnd_atk/sc_th_210/_index.tsv')
# aggregate_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/rnd_atk/sc_th_210/_stats.tsv')
# instance_type_names = {'0': 'Uniform'}
# index_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/rnd_atk/uniform/_index.tsv')
# aggregate_fpath = os.path.normpath('../Simulations/MN_nets/1cc_1ap/rnd_atk/uniform/_stats.tsv')

cols_to_ignore = ['no_sup_ccs', 'no_sup_relays', 'no_com_path']
# cols_to_ignore = None
run(index_fpath, aggregate_fpath, instance_type_names, cols_to_ignore)
