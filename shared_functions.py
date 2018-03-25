import os
import re
import sys
import shutil
import json
import logging.config
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

__author__ = 'Agostino Sturaro'


# to perform a natural sort, pass this function as the key to the sort functions
# sorted(list_of_strings, key=natural_sort_key)
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


# returns percentage value in [0.0, 1.0]
def percent_of_part(part, whole):
    if whole == 0:
        return 0.0
    return (1.0 * part) / whole


def percentage_split(seq, percentages):
    if sum(percentages) != 1.0:
        raise ValueError("The sum of percentages is not 1")
    start_pos = 0
    ele_cnt = len(seq)
    cumul_perc = 0
    split_seqs = []
    for perc in percentages:
        cumul_perc += perc
        stop_pos = int(cumul_perc * ele_cnt)
        split_seqs.append(seq[start_pos:stop_pos])
        start_pos = stop_pos
    return split_seqs


# Taken from http://stackoverflow.com/a/3041990
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("Invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def setup_logging(
        log_conf_path='logging.json',
        log_level=logging.INFO,  # used in case config file can't be found
        env_key='LOG_CFG'
):
    # if the environment variable is set, the logging configuration path is picked from there regardless
    value = os.getenv(env_key, None)
    if value is not None:
        log_conf_path = value

    log_conf_path = os.path.normpath(log_conf_path)
    if os.path.exists(log_conf_path):
        with open(log_conf_path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=log_level)


# access the ith column of whatever data structure numpy created
def get_unnamed_numpy_col_as_list(numpy_struct, col_num):
    if numpy_struct.dtype.fields is not None:
        col_name = 'f' + str(col_num)
        col = numpy_struct[col_name]  # access structured array
        col = col.tolist()
    else:
        if numpy_struct.ndim <= 1:
            col = numpy_struct  # direct access (single number o uniform 1D array)
        else:
            col = numpy_struct[:, col_num]  # access uniform multi-dimensional array
            col = col.tolist()
    # NumPy .tolist() might have returned a single element instead of a list, fix this
    if not isinstance(col, list):
        col = [col]
    return col


# compare two text files line by line, ignoring differences due to OS-specific newlines and
# the number of newlines at the end of the files
def compare_files_by_line(fpath1, fpath2, silent=True):
    with open(fpath1, 'r') as file1, open(fpath2, 'r') as file2:
        file1_end = False
        file2_end = False
        found_diff = False
        while not file1_end and not file2_end and not found_diff:
            try:
                f1_line = next(file1).rstrip('\r\n')  # strip '\n' and '\r' characters at the end of the line
            except StopIteration:
                f1_line = None
                file1_end = True
            try:
                f2_line = next(file2).rstrip('\r\n')
            except StopIteration:
                f2_line = None
                file2_end = True

            if f1_line != f2_line:
                if file1_end or file2_end:
                    if not (f1_line == '' or f2_line == ''):
                        found_diff = True
                        break
                else:
                    found_diff = True
                    break
    if found_diff is True and silent is False:
        print('Found difference in files, file 1 line:\n{}\nFile 2 line:\n{}'.format(f1_line, f2_line))
    return not found_diff


def is_graph_equal(G1, G2, data=False):
    if G1.is_directed() != G2.is_directed():
        return False

    if G1.is_multigraph() != G2.is_multigraph():
        return False

    if G1.name != G2.name:
        return False

    if data is True:
        if G1.graph != G2.graph:
            return False

    if G1.number_of_nodes() != G2.number_of_nodes():
        return False

    nodes_1 = sorted(G1.nodes(data=data))
    nodes_2 = sorted(G2.nodes(data=data))
    for idx in range(0, len(nodes_1)):
        # if the nodes have a data dict, this also compares its keys and values
        if nodes_1[idx] != nodes_2[idx]:
            return False

    if G1.number_of_edges() != G2.number_of_edges():
        return False

    # NetworkX can find edge differences quite easily
    edge_diff_graph = nx.symmetric_difference(G1, G2)
    edge_diff = edge_diff_graph.edges(data=False)
    if len(edge_diff) > 0:
        return False

    # two undirected graphs can have the same edges represented differently, e.g. (1, 2) vs (2, 1),
    # so we ask the edge data to NetworkX, which handles those differences
    if data is True:
        edges_1 = sorted(G1.edges(data=False))
        for u, v in edges_1:
            if G1.get_edge_data(u, v) != G2.get_edge_data(u, v):
                return False

    return True


def graph_diff(G1, G2, data=False):
    diff = ''

    if G1.is_directed() != G2.is_directed():
        diff += 'Different types of graphs\n'
        return diff

    if G1.is_multigraph() != G2.is_multigraph():
        diff += 'Different types of graphs\n'
        return diff

    if G1.name != G2.name:
        diff += 'Graphs have different names\n'
        diff += 'G1 name: {}\n'.format(G1.name)
        diff += 'G2 name: {}\n'.format(G2.name)

    if data is True:
        if G1.graph != G2.graph:
            diff += 'Graphs have different data\n'
            diff += 'G1 data: {}\n'.format(G1.graph)
            diff += 'G2 data: {}\n'.format(G2.graph)

    node_cnt_1 = G1.number_of_nodes()
    node_cnt_2 = G2.number_of_nodes()
    if node_cnt_1 != node_cnt_2:
        diff += 'Graphs have different node counts\n'
        diff += 'G1 node count: {}\n'.format(node_cnt_1)
        diff += 'G2 node count: {}\n'.format(node_cnt_2)

    nodes_1 = set(G1.nodes(data=False))
    nodes_2 = set(G2.nodes(data=False))

    node_diff_1 = nodes_1 - nodes_2
    if len(node_diff_1) > 0:
        node_diff_1 = sorted(node_diff_1)
        diff += 'Nodes exclusive to G1\n'
        for node in node_diff_1:
            diff += '{}\n'.format(node)

    node_diff_2 = nodes_2 - nodes_1
    if len(node_diff_2) > 0:
        node_diff_2 = sorted(node_diff_2)
        diff += 'Nodes exclusive to G2\n'
        for node in node_diff_2:
            diff += '{}\n'.format(node)

    # compare node data only if the graphs have the same nodes
    if data is True and len(node_diff_1) == 0 and len(node_diff_2) == 0:
        nodes_1 = sorted(G1.nodes(data=True))
        nodes_2 = sorted(G2.nodes(data=True))
        for idx in range(0, len(nodes_1)):
            if nodes_1[idx] != nodes_2[idx]:
                diff += 'Node {} has different data\n'.format(nodes_1[idx][0])
                diff += 'in G1: {}\n'.format(nodes_1[idx][1])
                diff += 'in G2: {}\n'.format(nodes_2[idx][1])

    edge_cnt_1 = G1.number_of_edges()
    edge_cnt_2 = G2.number_of_edges()
    if edge_cnt_1 != edge_cnt_2:
        diff += 'Graphs have different edge counts\n'
        diff += 'G1 edge count: {}\n'.format(edge_cnt_1)
        diff += 'G2 edge count: {}\n'.format(edge_cnt_2)

    # edge diff only works if the graphs have the same nodes
    if len(node_diff_1) == 0 and len(node_diff_2) == 0:
        edge_diff_graph_1 = nx.difference(G1, G2)
        edge_diff_1 = edge_diff_graph_1.edges(data=False)
        if len(edge_diff_1) > 0:
            # if the graph is undirected, adopt this edge representation (1, 2) over this (2, 1)
            if G1.is_directed() is False:
                edge_diff_1 = [sorted(edge) for edge in edge_diff_1]
            edge_diff_1 = sorted(edge_diff_1)
            diff += 'Edges exclusive to G1\n'
            for edge in edge_diff_1:
                diff += '{}\n'.format(edge)

        edge_diff_graph_2 = nx.difference(G2, G1)
        edge_diff_2 = edge_diff_graph_2.edges(data=False)
        if len(edge_diff_2) > 0:
            # if the graph is undirected, adopt this edge representation (1, 2) over this (2, 1)
            if G2.is_directed() is False:
                edge_diff_2 = [sorted(edge) for edge in edge_diff_2]
            edge_diff_2 = sorted(edge_diff_2)
            diff += 'Edges exclusive to G2\n'
            for edge in edge_diff_2:
                diff += '{}\n'.format(edge)

        # compare edge data only if the graphs have the same edges
        if data is True and len(edge_diff_1) == 0 and len(edge_diff_2) == 0:
            edges_1 = G1.edges(data=False)
            if G1.is_directed() is False:
                edges_1 = [sorted(edge) for edge in edges_1]
            edges_1 = sorted(edges_1)
            for u, v in edges_1:
                # two undirected graphs can have the same edges represented differently, e.g. (1, 2) vs (2, 1),
                # so we ask the edge data to NetworkX, which handles those differences
                edge_data_1 = G1.get_edge_data(u, v)
                edge_data_2 = G2.get_edge_data(u, v)
                if edge_data_1 != edge_data_2:
                    diff += 'Edge ({}, {}) has different data\n'.format(u, v)
                    diff += 'in G1: {}\n'.format(edge_data_1)
                    diff += 'in G2: {}\n'.format(edge_data_2)

    return diff


def compare_link_pos(G1, G2):
    if G1.is_directed() != G2.is_directed():
        return 'Cannot compare directed and undirected graphs'

    if G1.is_directed() is False:
        return 'Comparison between undirected graphs not implemented yet'

    diff = ''
    edges_0 = G1.edges(data=False)
    edges_1 = G2.edges(data=False)

    edge_cnt_0 = len(edges_0)
    edge_cnt_1 = len(edges_1)
    if edge_cnt_0 != edge_cnt_1:
        diff += 'Graphs have different edge counts\n'
        diff += 'G1 edge count: {}\n'.format(edge_cnt_0)
        diff += 'G2 edge count: {}\n'.format(edge_cnt_1)

    edge_pos_0 = []
    for edge in edges_0:
        src_node = G1.node[edge[0]]
        dst_node = G1.node[edge[1]]
        pos_arc = (src_node['x'], src_node['y'], dst_node['x'], dst_node['y'])
        edge_pos_0.append(pos_arc)

    edge_pos_1 = []
    for edge in edges_1:
        src_node = G2.node[edge[0]]
        dst_node = G2.node[edge[1]]
        pos_arc = (src_node['x'], src_node['y'], dst_node['x'], dst_node['y'])
        edge_pos_1.append(pos_arc)

    edge_pos_0 = sorted(edge_pos_0)
    edge_pos_1 = sorted(edge_pos_1)

    for idx in range(0, edge_cnt_0):
        edge_0 = edge_pos_0[idx]
        edge_1 = edge_pos_1[idx]
        if edge_0 != edge_1:
            diff += 'edge_0: {}\nedge_1: {}\n'.format(edge_0, edge_1)

    return diff


def paint_netw_graphs(A, B, Inter, node_col_by_role, edges_a_col, edges_b_col, x_shift_a=0.0, y_shift_a=0.0,
                      x_shift_b=0.0, y_shift_b=0.0, stretch=1.0, draw_labels=False, draw_nodes_kwargs={},
                      draw_edges_kwargs={}, draw_labels_kwargs={}):
    # remove the arguments we are going to override for the function draw_networkx_nodes
    if len(draw_nodes_kwargs) > 0:
        draw_nodes_kwargs.pop('G', None)
        draw_nodes_kwargs.pop('pos', None)
        draw_nodes_kwargs.pop('node_color', None)

    # remove the arguments we are going to override for the function draw_networkx_edges
    if len(draw_edges_kwargs) > 0:
        draw_edges_kwargs.pop('G', None)
        draw_edges_kwargs.pop('pos', None)
        draw_edges_kwargs.pop('edge_color', None)

    all_node_pos = dict()

    for node_a in A.nodes():
        all_node_pos[node_a] = (A.node[node_a]['x'] + x_shift_a, A.node[node_a]['y'] + y_shift_a)

    for node_b in B.nodes():
        all_node_pos[node_b] = (B.node[node_b]['x'] + x_shift_b, B.node[node_b]['y'] + y_shift_b)

    # spread all nodes over a wider area
    for node in all_node_pos:
        all_node_pos[node] = (all_node_pos[node][0] * stretch, all_node_pos[node][1] * stretch)

    # draw intra edges
    nx.draw_networkx_edges(A, all_node_pos, edge_color=edges_a_col, **draw_edges_kwargs)
    nx.draw_networkx_edges(B, all_node_pos, edge_color=edges_b_col, **draw_edges_kwargs)

    # draw inter edges, their role is determined by the role of their target node
    inter_edges_by_role = defaultdict(list)
    for edge in Inter.edges():
        target_node = edge[1]
        target_node_network = Inter.node[target_node]['network']
        if target_node_network == A.graph['name']:
            target_node_role = A.node[target_node]['role']
        else:
            target_node_role = B.node[target_node]['role']
        inter_edges_by_role[target_node_role].append(edge)

    for edge_role in inter_edges_by_role:
        nx.draw_networkx_edges(Inter, all_node_pos, edgelist=inter_edges_by_role[edge_role],
                               edge_color=node_col_by_role[edge_role], **draw_edges_kwargs)

    # draw nodes and node labels of A
    node_cols_a = list()
    for node_a in A.nodes():
        node_a_role = A.node[node_a]['role']
        node_cols_a.append(node_col_by_role[node_a_role])
    nx.draw_networkx_nodes(A, all_node_pos, node_color=node_cols_a, **draw_nodes_kwargs)

    # draw nodes and node labels of B
    node_cols_b = list()
    for node_b in B.nodes():
        node_b_role = B.node[node_b]['role']
        node_cols_b.append(node_col_by_role[node_b_role])
    nx.draw_networkx_nodes(B, all_node_pos, node_color=node_cols_b, **draw_nodes_kwargs)

    if draw_labels is True:
        nx.draw_networkx_labels(A, all_node_pos, **draw_labels_kwargs)
        nx.draw_networkx_labels(B, all_node_pos, **draw_labels_kwargs)


def paint_netw_graph(G, original_G, col_by_role, edge_col, pos_shifts=None, zoom=1, clear=False):
    if clear is True:
        plt.clf()

    # categorize nodes by vitality and group them by role

    alive_nodes = list()
    dead_nodes = list()

    for node in original_G.nodes():
        # role = original_G.node[node]['role']
        if G.has_node(node) is True:
            alive_nodes.append(node)
        else:
            dead_nodes.append(node)

    # categorize edges by vitality

    alive_edges = list()
    dead_edges = list()

    for edge in original_G.edges():
        if G.has_edge(*edge) is True:
            alive_edges.append(edge)
        else:
            dead_edges.append(edge)

    # calculate positions of all nodes

    alive_node_pos = dict()
    dead_node_pos = dict()

    if pos_shifts is None:
        for node in alive_nodes:
            alive_node_pos[node] = (G.node[node]['x'], G.node[node]['y'])
        for node in dead_nodes:
            dead_node_pos[node] = (original_G.node[node]['x'], original_G.node[node]['y'])
    else:
        for node in alive_nodes:
            alive_node_pos[node] = (G.node[node]['x'] + pos_shifts['x'],
                                    G.node[node]['y'] + pos_shifts['y'])
        for node in dead_nodes:
            dead_node_pos[node] = (original_G.node[node]['x'] + pos_shifts['x'],
                                   original_G.node[node]['y'] + pos_shifts['y'])

    all_node_pos = alive_node_pos.copy()
    all_node_pos.update(dead_node_pos)

    # spread all nodes over a wider area - HACK
    for key in all_node_pos:
        all_node_pos[key] = (all_node_pos[key][0] * zoom, all_node_pos[key][1] * zoom)

    # draw edges
    nx.draw_networkx_edges(G, all_node_pos, edgelist=alive_edges, edge_color=edge_col, alpha=0.7)
    nx.draw_networkx_edges(original_G, all_node_pos, edgelist=dead_edges, edge_color='gray', alpha=0.7)

    # decide node colors

    node_cols = list()
    for node in alive_nodes:
        node_role = G.node[node]['role']
        node_cols.append(col_by_role[node_role])

    for _ in dead_nodes:
        node_cols.append('gray')

    # draw nodes
    nx.draw_networkx_nodes(G, all_node_pos, nodelist=alive_nodes + dead_nodes, node_size=100,
                           node_color=node_cols, alpha=0.7)

    # draw node labels
    nx.draw_networkx_labels(original_G, all_node_pos, font_size=1)


# TODO: STOP READING NODE INFORMATION FROM THE INTER GRAPH AND READ FROM THE ACTUAL NET GRAPHS
# same function as the one used in netw_export_test
def paint_inter_graph(G, original_G, edge_col, edge_col_per_type, pos_shifts_by_netw, zoom=1.0):
    # categorize edges by vitality

    # alive_edges = list()
    # dead_edges = list()
    #
    # for edge in original_G.edges():
    #     if G.has_edge(*edge) is True:
    #         alive_edges.append(edge)
    #     else:
    #         dead_edges.append(edge)

    alive_edges_per_type = defaultdict(list)
    dead_edges_per_type = defaultdict(list)

    for edge in original_G.edges():
        edge_type = original_G.node[edge[1]]['role']

        if G.has_edge(*edge) is True:
            alive_edges_per_type[edge_type].append(edge)
        else:
            dead_edges_per_type[edge_type].append(edge)

    # calculate positions of all nodes

    all_node_pos = dict()

    for node in original_G.nodes():
        node_netw = original_G.node[node]['network']
        all_node_pos[node] = (original_G.node[node]['x']
                              + pos_shifts_by_netw[node_netw]['x'],
                              original_G.node[node]['y']
                              + pos_shifts_by_netw[node_netw]['y'])

    # spread all nodes over a wider area - HACK
    for key in all_node_pos:
        all_node_pos[key] = (all_node_pos[key][0] * zoom, all_node_pos[key][1] * zoom)

    # draw edges
    for edge_type in alive_edges_per_type:
        nx.draw_networkx_edges(G, all_node_pos, edgelist=alive_edges_per_type[edge_type], width=0.2,
                               edge_color=edge_col_per_type[edge_type], arrows=True, alpha=0.7)
    for edge_type in dead_edges_per_type:
        nx.draw_networkx_edges(original_G, all_node_pos, edgelist=dead_edges_per_type[edge_type], width=0.2,
                               edge_color='gray', arrows=True, alpha=0.7)


def mix(colors, markers, desired_cnt=None):
    color_cnt = len(colors)
    marker_cnt = len(markers)

    if desired_cnt is None:
        desired_cnt = color_cnt * marker_cnt
    elif desired_cnt > color_cnt * marker_cnt:
        sys.exit("Cannot produce more couples than len(colors) * len(markers)")

    couples = []
    counter = 0
    if color_cnt < marker_cnt:
        while counter < desired_cnt:
            couples.append((markers[counter % marker_cnt],
                            colors[(counter % color_cnt + (counter / color_cnt)) % color_cnt]))
            counter += 1
    else:
        while counter < desired_cnt:
            couples.append((markers[(counter % marker_cnt + (counter / marker_cnt)) % marker_cnt],
                            colors[counter % color_cnt]))
            counter += 1
    return couples


def makedirs_clean(path, clean_subdirs=False, ask_confirmation=False):
    path = os.path.normpath(path)

    if not os.path.exists(path):
        # print("dir does not exist")
        os.makedirs(path)
    else:
        fnames = os.listdir(path)

        # if the directory exists, but is empty, do nothing
        if len(fnames) == 0:
            return

        # otherwise
        if ask_confirmation is False or query_yes_no('Output directory not empty, do you want to remove its files?\n'
                                                     'Path: ' + str(path)) is True:
            # remove all files in the directory, and optionally remove its subdirectories
            for fname in fnames:
                fpath = os.path.join(path, fname)
                if os.path.isfile(fpath) is True:
                    os.remove(fpath)
                elif clean_subdirs is True:
                    shutil.rmtree(fpath)


def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
