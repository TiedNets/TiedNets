__author__ = 'sturaroa'

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import shared_functions as sf

from PyPDF2 import PdfFileMerger, PdfFileReader

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0

config = ConfigParser()
conf_path = os.path.normpath('C:/Users/sturaroa/Documents/Simulations/exp_1000n_test_2/rnd_atk/realistic/instance_0/run_0.ini')
config.read(conf_path)

# copypaste
area_size = 5
margin = area_size * 0.02
dist_perc = 0.16

base_graphs_dir = os.path.normpath(config.get('paths', 'netw_dir'))
step_graphs_dir = os.path.normpath(config.get('paths', 'results_dir'))
netw_a_fname = config.get('paths', 'netw_a_fname')
netw_b_fname = config.get('paths', 'netw_b_fname')
netw_inter_fname = config.get('paths', 'netw_inter_fname')
steps_index_fname = config.get('paths', 'run_stats_fname')
times = []

# opened in text-mode; all EOLs are converted to '\n'
steps_index = os.path.normpath(os.path.join(step_graphs_dir, steps_index_fname))

# open file skipping the first line, then read values by column
my_data = np.genfromtxt(steps_index, delimiter='\t', skip_header=1, dtype=None)
times = sf.get_unnamed_numpy_col(my_data, 0)

# read base graphs
original_A = nx.read_graphml(os.path.join(base_graphs_dir, netw_a_fname))
original_B = nx.read_graphml(os.path.join(base_graphs_dir, netw_b_fname))
original_I = nx.read_graphml(os.path.join(base_graphs_dir, netw_inter_fname))

# map used to separate nodes of the 2 networks (e.g. draw A nodes on the left side and B nodes on the right)
pos_shifts_by_netw = {original_A.graph['name']: {'x': 0, 'y': 0},
                      original_B.graph['name']: {'x': area_size + area_size * dist_perc, 'y': 0}}

print('times ' + str(times))  # debug

# draw graphs for eachs step
pdf_fpaths = []
for time in times:
    print('time ' + str(time))
    # copypaste
    plt.figure(figsize=(15 + 1.6, 10))
    plt.xlim(-margin, area_size * 2 + area_size * dist_perc + margin)
    plt.ylim(-margin, area_size + margin)

    A = nx.read_graphml(os.path.join(step_graphs_dir, str(time) + '_' + netw_a_fname))
    sf.paint_netw_graph(A, original_A, {'power': 'r', 'generator': 'r', 'transmission_substation': 'plum',
                                        'distribution_substation': 'magenta'}, 'r')

    B = nx.read_graphml(os.path.join(step_graphs_dir, str(time) + '_' + netw_b_fname))
    sf.paint_netw_graph(B, original_B, {'communication': 'b', 'controller': 'c', 'relay': 'b'}, 'b',
                        pos_shifts_by_netw[B.graph['name']])

    I = nx.read_graphml(os.path.join(step_graphs_dir, str(time) + '_' + netw_inter_fname))

    edge_col_per_type = {'power': 'r', 'generator': 'r', 'transmission_substation': 'plum',
                     'distribution_substation': 'magenta', 'communication': 'b', 'controller': 'c', 'relay': 'b'}
    # sf.paint_inter_graph(I, original_I, 'orange', pos_shifts_by_netw, edge_col_per_type)

    pdf_fpaths.append(os.path.join(step_graphs_dir, str(time) + '_full.pdf'))
    plt.savefig(pdf_fpaths[-1])  # get last element in list
    # plt.show()
    plt.close()  # free memory

# merge produced pdf files
merger = PdfFileMerger()
merger.append(file(os.path.join(base_graphs_dir, '_full.pdf'), 'rb'))
for fpath in pdf_fpaths:
    merger.append(file(fpath, 'rb'))  # best way to avoid keeping files open
merger.write(os.path.join(step_graphs_dir, '_merge.pdf'))
merger.close()  # free memory

# remove partial files
for fpath in pdf_fpaths:
    os.remove(fpath)
