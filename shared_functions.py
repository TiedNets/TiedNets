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


def paint_netw_graphs(A, B, Inter, node_col_by_role, edges_a_col, edges_b_col, x_shift_a=0.0, y_shift_a=0.0,
                      x_shift_b=0.0, y_shift_b=0.0, stretch=1.0, font_size=5.0, draw_nodes_kwargs={},
                      draw_edges_kwargs={}):
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
    nx.draw_networkx_labels(A, all_node_pos, font_size=font_size)

    # draw nodes and node labels of B
    node_cols_b = list()
    for node_b in B.nodes():
        node_b_role = B.node[node_b]['role']
        node_cols_b.append(node_col_by_role[node_b_role])
    nx.draw_networkx_nodes(B, all_node_pos, node_color=node_cols_b, **draw_nodes_kwargs)
    nx.draw_networkx_labels(B, all_node_pos, font_size=font_size)


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
    # nx.draw_networkx_edges(original_G, all_node_pos, edgelist=dead_edges, edge_color='gray', alpha=0.7)

    # decide node colors

    node_cols = list()
    for node in alive_nodes:
        node_role = G.node[node]['role']
        node_cols.append(col_by_role[node_role])

    for node in dead_nodes:
        node_cols.append('gray')

    # draw nodes
    # nx.draw_networkx_nodes(G, all_node_pos, nodelist=alive_nodes + dead_nodes, node_size=100,
    #                        node_color=node_cols, alpha=0.7)
    nx.draw_networkx_nodes(G, all_node_pos, nodelist=alive_nodes, node_size=5,
                           node_color=node_cols, alpha=0.7, linewidths=0.0)
    # nx.draw_networkx_nodes(G, all_node_pos, nodelist=dead_nodes, node_size=5,
    #                        node_color='gray', alpha=0.7)

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
