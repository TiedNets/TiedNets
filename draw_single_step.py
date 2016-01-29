__author__ = 'Agostino Sturaro'

import os
import networkx as nx
import matplotlib.pyplot as plt
import shared_functions as sf

def arrange_nodes(G, pos_by_node):
    for node, (x, y) in pos_by_node.items():
        G.node[node]['x'] = float(x)
        G.node[node]['y'] = float(y)

output_dir = 'C:/Users/Agostino/Documents/Simulations/MN/rnd_atk/uniform/instance_0/run_1/'

original_A = nx.read_graphml('C:/Users/Agostino/Documents/Simulations/MN/instance_0/A.graphml',
                             node_type=str)
A = nx.read_graphml('C:/Users/Agostino/Documents/Simulations/MN/rnd_atk/uniform/instance_0/run_1/18_A.graphml',
                    node_type=str)
original_B = nx.read_graphml('C:/Users/Agostino/Documents/Simulations/MN/instance_0/B.graphml',
                             node_type=str)
B = nx.read_graphml('C:/Users/Agostino/Documents/Simulations/MN/rnd_atk/uniform/instance_0/run_1/18_B.graphml',
                    node_type=str)
# I = nx.read_graphml('C:/Users/Agostino/Documents/Simulations/MN/rnd_atk/uniform/instance_0/run_0/22_InterMM.graphml',
#                     node_type=str)

# hack, use only if nodes do not have x,y
# hack, assuming a lot of values are the same
span = 1.0
pos_by_node = nx.spring_layout(A, dim=2, scale=span)
arrange_nodes(A, pos_by_node)
pos_by_node = nx.spring_layout(B, dim=2, scale=span)
arrange_nodes(B, pos_by_node)

netw_a_name = A.graph['name']
netw_b_name = B.graph['name']

first_it = True
for node in B.nodes():
    node_inst = B.node[node]
    x = node_inst['x']
    y = node_inst['y']

    if first_it is True:
        x_min = x
        y_min = y
        x_max = x
        y_max = y
        first_it = False
    else:
        if x > x_max:
            x_max = x
        elif x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        elif y < y_min:
            y_min = y

span = max(x_max - x_min, y_max - y_min)

# draw networks
dist_perc = 0.16
plt.figure(figsize=(15 + 1.6, 10))

# map used to separate nodes of the 2 networks (e.g. draw A nodes on the left side and B nodes on the right)
# pos_shifts_by_netw = {netw_a_name: {'x': 0, 'y': 0},
#                       netw_b_name: {'x': span + span * dist_perc, 'y': 0}}
pos_shifts_by_netw = {netw_a_name: {'x': 0, 'y': 0},
                      netw_b_name: {'x': 0, 'y': 0}}

edge_col_per_type = {'power': 'r', 'generator': 'r', 'transmission_substation': 'plum',
                     'distribution_substation': 'magenta', 'communication': 'b', 'controller': 'c', 'relay': 'b'}
sf.paint_netw_graph(A, original_A, edge_col_per_type, 'r')
sf.paint_netw_graph(B, original_B, edge_col_per_type, 'b', pos_shifts_by_netw[netw_b_name])

# sf.paint_inter_graph(I, I, 'orange', pos_shifts_by_netw, edge_col_per_type)

plt.savefig(os.path.join(output_dir, '_finish.pdf'))
# plt.show()
plt.close()  # free memory
