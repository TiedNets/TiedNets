__author__ = 'Agostino Sturaro'

import os
import re
import sys
import shutil
import json
import logging.config
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


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
    log_conf_path = os.path.normpath(log_conf_path)
    if os.path.isabs(log_conf_path) is False:
        this_dir = os.path.dirname(__file__)
        log_conf_path = os.path.join(this_dir, log_conf_path)

    value = os.getenv(env_key, None)
    if value:
        log_conf_path = value
    if os.path.exists(log_conf_path):
        with open(log_conf_path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=log_level)


# access the ith column of whatever data structure numpy created
def get_unnamed_numpy_col(numpy_struct, col_num):
    if numpy_struct.dtype.fields is not None:
        col_name = 'f' + str(col_num)
        col = numpy_struct[col_name]  # access structured array
    else:
        if numpy_struct.ndim <= 1:
            col = numpy_struct  # direct access (single number o uniform 1D array)
        else:
            col = numpy_struct[:, col_num]  # access uniform multi-dimensional array

    return col


def paint_netw_graph(G, original_G, col_by_role, edge_col, pos_shifts=None, clear=False):
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
                           node_color=node_cols, alpha=0.7)
    # nx.draw_networkx_nodes(G, all_node_pos, nodelist=dead_nodes, node_size=5,
    #                        node_color='gray', alpha=0.7)

    # draw node labels
    # nx.draw_networkx_labels(original_G, all_node_pos, font_size=6)


# TODO: STOP READING NODE INFORMATION FROM THE INTER GRAPH AND READ FROM THE ACTUAL NET GRAPHS
# same function as the one used in netw_export_test
def paint_inter_graph(G, original_G, edge_col, pos_shifts_by_netw, edge_col_per_type):
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

    # draw edges
    for edge_type in alive_edges_per_type:
        nx.draw_networkx_edges(G, all_node_pos, edgelist=alive_edges_per_type[edge_type], style='dashed',
                               edge_color=edge_col_per_type[edge_type], arrows=False, alpha=0.7)
    for edge_type in dead_edges_per_type:
        nx.draw_networkx_edges(original_G, all_node_pos, edgelist=dead_edges_per_type[edge_type],
                               edge_color='gray', arrows=False, alpha=0.7)


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
            # print("dir exists but is empty")
            return

        # print("dir exists and is not empty")
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
