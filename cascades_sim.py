import os
import logging
import networkx as nx
import random
import csv
import sys
import shared_functions as sf
from numpy import percentile

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0

__author__ = 'Agostino Sturaro'

# global variables
logger = logging.getLogger(__name__)
time = None


# choose node_cnt random nodes
def choose_random_nodes(G, node_cnt, seed=None):
    candidates = G.nodes()
    my_random = random.Random(seed)
    my_random.shuffle(candidates)
    chosen_nodes = list()
    for i in range(0, node_cnt):
        chosen_nodes.append(candidates[i])
    return chosen_nodes


def choose_random_nodes_except(G, node_cnt, excluded, seed=None):
    candidates = G.nodes()
    my_random = random.Random(seed)
    my_random.shuffle(candidates)
    chosen_nodes = list()
    idx = 0
    while len(chosen_nodes) < node_cnt:
        node = candidates[idx]
        if node not in excluded:
            chosen_nodes.append(node)
        idx += 1
    return chosen_nodes


def pick_ith_node(G, node_rank):
    nodes = G.nodes()
    nodes.sort()
    return nodes[node_rank]


# ranked_nodes is a list of nodes, which can be from both A and B, sorted in a deterministic way
# [node with lowest rank, ..., node with highest rank]
# node_cnt is the number of nodes to pick
# from_netw is the name of the network we want to pick nodes from
# I is the graph of interconnections between A and B
# bottom_skips is the number of ranks to skip from the bottom ranks, to avoid picking nodes with those ranks
def pick_nodes_by_rank_from_bottom(ranked_nodes, node_cnt, from_netw, I, bottom_skips):
    chosen_nodes = list()
    if from_netw == 'both':
        for i in range(bottom_skips, bottom_skips + node_cnt):
            node = ranked_nodes[i]
            chosen_nodes.append(node)
    else:
        idx = bottom_skips
        while len(chosen_nodes) < node_cnt:
            node = ranked_nodes[idx]
            node_netw = I.node[node]['network']
            if node_netw == from_netw:
                chosen_nodes.append(node)
            idx += 1

    return chosen_nodes


def pick_nodes_by_rank_from_top(ranked_nodes, node_cnt, from_netw, I, top_skips):
    chosen_nodes = list()
    top_pos = len(ranked_nodes) - 1
    start_pos = top_pos - top_skips
    if from_netw == 'both':
        for i in range(start_pos, start_pos - node_cnt, -1):
            node = ranked_nodes[i]
            chosen_nodes.append(node)
    else:
        idx = start_pos
        while len(chosen_nodes) < node_cnt:
            node = ranked_nodes[idx]
            node_netw = I.node[node]['network']
            if node_netw == from_netw:
                chosen_nodes.append(node)
            idx -= 1

    return chosen_nodes


def pick_random_nodes_in_rank_range(ranked_nodes, node_cnt, from_netw, I, bottom_skips, top_skips, seed=None):
    my_random = random.Random(seed)
    chosen_nodes = list()
    nodes_in_range = ranked_nodes[bottom_skips:-top_skips]
    my_random.shuffle(nodes_in_range)
    if from_netw == 'both':
        for i in range(0, node_cnt):
            chosen_nodes.append(nodes_in_range[i])
    else:
        idx = 0
        while len(chosen_nodes) < node_cnt:
            node = nodes_in_range[idx]
            node_netw = I.node[node]['network']
            if node_netw == from_netw:
                chosen_nodes.append(node)
            idx += 1

    return chosen_nodes


# score_node_pairs is a list of tuples (score, node), nodes with the same score are sorted using their id
def pick_nodes_by_score(score_node_pairs, node_cnt):
    if len(score_node_pairs) < node_cnt:
        raise RuntimeError("Insufficient nodes to select")

    # sort the data structure by rank first and then by node id (tuple sorting)
    sorted_pairs = sorted(score_node_pairs)

    chosen_nodes = list()
    for i in range(0, node_cnt):
        chosen_nodes.append(sorted_pairs.pop()[1])

    return chosen_nodes


def choose_most_inter_used_nodes(G, I, node_cnt, role):
    # select nodes with the specified role from graph G and create a list of tuples with their in-degree in graph I
    rank_node_pairs = list()
    if role == 'any':
        for node in G.nodes():
            if nx.is_directed(I):
                rank = I.in_degree(node)
            else:
                rank = I.degree(node)
            rank_node_pairs.append((rank, node))
    else:
        for node in G.nodes():
            node_role = G.node[node]['role']
            if node_role == role:
                if nx.is_directed(I):
                    rank = I.in_degree(node)
                else:
                    rank = I.degree(node)
                rank_node_pairs.append((rank, node))

    return pick_nodes_by_score(rank_node_pairs, node_cnt)


def choose_most_intra_used_nodes(G, node_cnt, role):
    # select nodes from G, get their degree
    # make that a list of tuples
    rank_node_pairs = list()
    if role == 'any':
        for node in G.nodes():
            rank = G.degree(node)
            rank_node_pairs.append((rank, node))
    else:
        for node in G.nodes():
            node_role = G.node[node]['role']
            if node_role == role:
                rank = G.degree(node)
                rank_node_pairs.append((rank, node))

    return pick_nodes_by_score(rank_node_pairs, node_cnt)


# find nodes that are not in the giant component, used by the basic models
def find_nodes_not_in_giant_component(G):
    unsupported_nodes = list()
    components = sorted(nx.connected_components(G), key=len, reverse=True)

    for component_idx in range(1, len(components)):
        for node in components[component_idx]:
            unsupported_nodes.append(node)

    return unsupported_nodes


# find nodes that have no inter-link, used by the basic models and by the realistic model
def find_nodes_without_inter_links(G, I):
    unsupported_nodes = list()

    for node in G.nodes():
        if I.has_node(node):
            supporting_nodes = I.neighbors(node)
            if len(supporting_nodes) <= 0:
                unsupported_nodes.append(node)
        else:
            unsupported_nodes.append(node)

    return unsupported_nodes


# find nodes that are in clusters with size smaller than the minimum size
def find_nodes_in_smaller_clusters(G, min_cluster_size):
    unsupported_nodes = list()
    components = sorted(nx.connected_components(G), key=len, reverse=True)

    for component in components:
        if len(component) < min_cluster_size:
            for node in component:
                unsupported_nodes.append(node)

    return unsupported_nodes


# find nodes in small clusters without any inter links, not used
def find_nodes_in_unsupported_clusters(G, I):
    unsupported_nodes = list()
    components = sorted(nx.connected_components(G), key=len, reverse=True)

    for component in components:
        support_found = False
        for node in component:
            if I.has_node(node):
                if len(I.neighbors(node)) > 0:
                    support_found = True
                    break
        if support_found is False:
            for node in component:
                unsupported_nodes.append(node)

    return unsupported_nodes


# find substation nodes in the power network that are not connected to a generator, used by the realistic model
def find_unpowered_substations(G):
    global logger
    unsupported_nodes = list()

    # divide nodes by role
    generators = list()
    substations = list()
    for node in G.nodes():
        role = G.node[node]['role']
        if role == 'generator':
            generators.append(node)
        elif role in ['transmission_substation', 'distribution_substation']:
            substations.append(node)

    # find out which substations are not powered by a generator
    for substation in substations:
        support_found = False
        for generator in generators:
            if nx.has_path(G, substation, generator):
                support_found = True
                break
        if support_found is False:
            unsupported_nodes.append(substation)

    return unsupported_nodes


# find nodes of the power grid that have no access to a control center, used by the realistic model
def find_uncontrolled_pow_nodes(A, B, I, by_reason=False):
    unsupported_nodes_by_reason = {'no_sup_ccs': [], 'no_sup_relays': [], 'no_com_path': []}

    # for each power node
    for node_a in A.nodes():
        support_controllers = list()
        support_relays = list()

        # find the nodes it's supported by in the other network (check the inter-graph) and separate them by role
        if I.has_node(node_a):
            for node_b in I.neighbors(node_a):
                role = B.node[node_b]['role']
                if role == 'relay':
                    support_relays.append(node_b)
                elif role == 'controller':
                    support_controllers.append(node_b)

        # check if there's a control node supporting the power node, then check if there's a relay node granting it
        # access to the communication network and, finally, check if one of the supporting control centers can be
        # reached from one of the supporting relay nodes
        support_found = False
        if len(support_controllers) < 1:
            unsupported_nodes_by_reason['no_sup_ccs'].append(node_a)
        else:
            if len(support_relays) < 1:
                unsupported_nodes_by_reason['no_sup_relays'].append(node_a)
            else:
                for controller in support_controllers:
                    for relay in support_relays:
                        if nx.has_path(B, controller, relay):
                            support_found = True
                            break
                    if support_found is True:
                        break
                if support_found is False:
                    unsupported_nodes_by_reason['no_com_path'].append(node_a)

    if by_reason is True:
        return unsupported_nodes_by_reason
    else:
        unsupported_nodes = list()
        for node_list in unsupported_nodes_by_reason.values():
            unsupported_nodes.extend(node_list)
        return unsupported_nodes


def save_state(time, A, B, I, results_dir):
    netw_a_fpath_out = os.path.join(results_dir, str(time) + '_' + A.graph['name'] + '.graphml')
    nx.write_graphml(A, netw_a_fpath_out)
    netw_b_fpath_out = os.path.join(results_dir, str(time) + '_' + B.graph['name'] + '.graphml')
    nx.write_graphml(B, netw_b_fpath_out)
    netw_inter_fpath_out = os.path.join(results_dir, str(time) + '_' + I.graph['name'] + '.graphml')
    nx.write_graphml(I, netw_inter_fpath_out)


# calculate the percentage of nodes attacked for each role
# result_key_by_role is a dictionary {name of the node role: key to use for the corresponding metric in the result}
def calc_atkd_percent_by_role(G, attacked_nodes, result_key_by_role):
    atkd_perc_by_role = {}

    node_cnt_by_role = {}
    node_roles = result_key_by_role.keys()
    for node_role in node_roles:
        node_cnt_by_role[node_role] = 0
    atkd_node_cnt_by_role = node_cnt_by_role.copy()

    for node in G.nodes():
        node_role = G.node[node]['role']
        node_cnt_by_role[node_role] += 1

    for node in attacked_nodes:
        node_role = G.node[node]['role']
        atkd_node_cnt_by_role[node_role] += 1

    for node_role in node_roles:
        result_name = result_key_by_role[node_role]
        atkd_perc_by_role[result_name] = sf.percent_of_part(atkd_node_cnt_by_role[node_role],
                                                            node_cnt_by_role[node_role])

    return atkd_perc_by_role


# For a given centrality, calculate:
# - p_q_{i}_{centrality_name} the fraction of nodes in quintile {i} that were attacked
# - p_sum_{centrality_name} the sum of the centrality scores of the attacked nodes
# - p_tot_{centrality_name} the fraction of total centrality score that was attacked
#
# NOTE: we consider a node to be "part of the 2nd quintile" if it "has a centrality score greater than the 1st quintile
# mark and lower than the 3rd quintile mark". Our definition might not be the same as the canonical definition, but it
# is what we need to use.
def calc_atk_centrality_stats(attacked_nodes, centrality_name, result_key_suffix, centrality_info):
    global logger
    centr_stats = {}  # dictionary used to index statistics with shorter names

    node_cnt = centrality_info['node_count']
    centr_by_node = centrality_info[centrality_name]

    # sum the centrality scores of the attacked nodes
    atkd_centr_sum = 0.0
    total_centr = centrality_info['total_' + centrality_name]
    for node in attacked_nodes:
        atkd_centr_sum += centr_by_node[node]
    stat_name = 'sum_' + result_key_suffix
    centr_stats[stat_name] = atkd_centr_sum
    stat_name = 'p_tot_' + result_key_suffix
    centr_stats[stat_name] = sf.percent_of_part(atkd_centr_sum, total_centr)

    # count how many nodes were attacked in each quintile
    quintiles = centrality_info[centrality_name + '_quintiles']
    atkd_cnt_by_quintile = [0] * 5
    quintiles_set = set(quintiles)
    if len(quintiles_set) == len(quintiles):
        for node in attacked_nodes:
            node_quintile = 0
            node_centr = centr_by_node[node]
            for quintile_val in quintiles:
                if node_centr > quintile_val:
                    node_quintile += 1
                else:
                    break
            atkd_cnt_by_quintile[node_quintile] += 1
    else:
        # this is a corner case presented by graphs in which many nodes have identical centrality scores, like in
        # random regular graphs. To determine what quintile each node belongs to, we check their position in our
        # deterministic ranking, where they are sorted by their centrality score first and then by their id
        # logger.info('There are only {} different quintiles for centrality {}'.format(len(quintiles_set),
        #                                                                              centrality_name))
        ranked_nodes = centrality_info[centrality_name + '_rank']
        # find the positions that separate the quintiles
        rank_of_quintiles = percentile(range(0, node_cnt), [20, 40, 60, 80]).tolist()
        logger.debug('rank_of_quintiles {}'.format(rank_of_quintiles))

        for node in attacked_nodes:
            node_quintile = 0
            node_rank = ranked_nodes.index(node)
            for quintile_rank in rank_of_quintiles:
                if node_rank > quintile_rank:
                    node_quintile += 1
                else:
                    break
            atkd_cnt_by_quintile[node_quintile] += 1

    logger.debug('atkd_cnt_by_quintile {}'.format(atkd_cnt_by_quintile))

    nodes_in_quintile = 1.0 * node_cnt / 5
    logger.debug('nodes_in_quintile {}'.format(nodes_in_quintile))

    # calculate the percentage of nodes attacked in each quintile
    atkd_perc_of_quintile = [0.0] * 5
    for i in range(0, 5):
        atkd_perc_of_quintile[i] = sf.percent_of_part(atkd_cnt_by_quintile[i], nodes_in_quintile)
    logger.debug('atkd_perc_of_quintile {}'.format(atkd_perc_of_quintile))

    # p_q_1 is fraction of nodes attacked in the first quintile
    for i, val in enumerate(atkd_perc_of_quintile):
        stat_name = 'p_q_{}_{}'.format(i + 1, result_key_suffix)
        centr_stats[stat_name] = val

    return centr_stats


def calc_atk_centr_stats(name_A, name_B, name_I, name_AB, attacked_nodes_a, attacked_nodes_b, floader, netw_dir):
    attacked_nodes = attacked_nodes_a + attacked_nodes_b
    centr_stats = {}

    # load files with precalculated centrality metrics
    centr_fpath_a = os.path.join(netw_dir, 'node_centrality_{}.json'.format(name_A))
    centr_info_a = floader.fetch_json(centr_fpath_a)
    centr_fpath_b = os.path.join(netw_dir, 'node_centrality_{}.json'.format(name_B))
    centr_info_b = floader.fetch_json(centr_fpath_b)
    centr_fpath_ab = os.path.join(netw_dir, 'node_centrality_{}.json'.format(name_AB))
    centr_info_ab = floader.fetch_json(centr_fpath_ab)
    centr_fpath_i = os.path.join(netw_dir, 'node_centrality_{}.json'.format(name_I))
    centr_info_i = floader.fetch_json(centr_fpath_i)
    centr_fpath_gen = os.path.join(netw_dir, 'node_centrality_misc.json')
    centr_info_misc = floader.fetch_json(centr_fpath_gen)

    node_cnt_a = centr_info_a['node_count']
    centr_stats['p_atkd_a'] = sf.percent_of_part(len(attacked_nodes_a), node_cnt_a)

    node_cnt_b = centr_info_b['node_count']
    centr_stats['p_atkd_b'] = sf.percent_of_part(len(attacked_nodes_b), node_cnt_b)

    centr_name = 'betweenness_centrality'
    centr_stats.update(calc_atk_centrality_stats(attacked_nodes_a, centr_name, 'atkd_betw_c_a', centr_info_a))
    centr_stats.update(calc_atk_centrality_stats(attacked_nodes_b, centr_name, 'atkd_betw_c_b', centr_info_b))
    centr_stats.update(calc_atk_centrality_stats(attacked_nodes, centr_name, 'atkd_betw_c_ab', centr_info_ab))
    centr_stats.update(calc_atk_centrality_stats(attacked_nodes, centr_name, 'atkd_betw_c_i', centr_info_i))

    centr_name = 'closeness_centrality'
    centr_stats.update(calc_atk_centrality_stats(attacked_nodes_a, centr_name, 'atkd_clos_c_a', centr_info_a))
    centr_stats.update(calc_atk_centrality_stats(attacked_nodes_b, centr_name, 'atkd_clos_c_b', centr_info_b))
    centr_stats.update(calc_atk_centrality_stats(attacked_nodes, centr_name, 'atkd_clos_c_ab', centr_info_ab))
    centr_stats.update(calc_atk_centrality_stats(attacked_nodes, centr_name, 'atkd_clos_c_i', centr_info_i))

    centr_name = 'degree_centrality'
    centr_stats.update(calc_atk_centrality_stats(attacked_nodes_a, centr_name, 'atkd_deg_c_a', centr_info_a))
    centr_stats.update(calc_atk_centrality_stats(attacked_nodes_b, centr_name, 'atkd_deg_c_b', centr_info_b))

    centr_name = 'indegree_centrality'
    if centr_name in centr_info_ab:
        centr_stats.update(calc_atk_centrality_stats(attacked_nodes, centr_name, 'atkd_indeg_c_ab', centr_info_ab))
    if centr_name in centr_info_i:
        centr_stats.update(calc_atk_centrality_stats(attacked_nodes, centr_name, 'atkd_indeg_c_i', centr_info_i))

    centr_name = 'katz_centrality'
    if centr_name in centr_info_ab:
        centr_stats.update(calc_atk_centrality_stats(attacked_nodes, centr_name, 'atkd_katz_c_ab', centr_info_ab))
    if centr_name in centr_info_i:
        centr_stats.update(calc_atk_centrality_stats(attacked_nodes, centr_name, 'atkd_katz_c_i', centr_info_i))

    centr_name = 'relay_betweenness_centrality'
    if centr_name in centr_info_misc:
        centr_info_misc['node_count'] = centr_info_misc['relay_count']  # trick to reuse the same logic
        centr_stats.update(calc_atk_centrality_stats(attacked_nodes, centr_name, 'atkd_rel_betw_c', centr_info_misc))

    centr_name = 'transm_subst_betweenness_centrality'
    if centr_name in centr_info_misc:
        centr_info_misc['node_count'] = centr_info_misc['transmission_substation_count']  # trick
        centr_stats.update(calc_atk_centrality_stats(attacked_nodes, centr_name, 'atkd_ts_betw_c', centr_info_misc))

    return centr_stats


def get_ranked_nodes(config, section, floader, netw_dir):
    bottom_skips = 0
    if config.has_option(section, 'bottom_ranks_to_skip'):
        bottom_skips = config.getint(section, 'bottom_ranks_to_skip')

    top_skips = 0
    if config.has_option(section, 'top_ranks_to_skip'):
        top_skips = config.getint(section, 'top_ranks_to_skip')

    # the name of the network file to pick centrality measures from
    centr_fname = config.get(section, 'centrality_fname')

    # load file with precalculated centrality metrics
    centr_fpath = os.path.join(netw_dir, centr_fname)
    centrality_info = floader.fetch_json(centr_fpath)

    # load the list of ranked nodes
    centrality_name = config.get(section, 'centrality_name')
    ranked_nodes = centrality_info[centrality_name + '_centrality_rank']

    return ranked_nodes, bottom_skips, top_skips


# either returns the same list or a filtered copy (not the best interface)
def remove_list_items(list_to_filter, items):
    if len(items) == 0 or len(list_to_filter) == 0:
        return list_to_filter
    else:
        return [item for item in list_to_filter if item not in items]


def remove_items_from_lists_in_dict(dict_of_lists, items):
    if len(items) > 0:
        for key in dict_of_lists:
            list_to_filter = dict_of_lists[key]
            list_to_filter = [item for item in list_to_filter if item not in items]
            dict_of_lists[key] = list_to_filter


def choose_nodes_by_config(config, section, target_netw, method, A, B, I, ab_union, floader, netw_dir, seed):
    if target_netw == A.graph['name']:
        target_G = A
    elif target_netw == B.graph['name']:
        target_G = B
    elif target_netw.lower() == 'both':
        if ab_union is None:
            raise ValueError('A union graph is needed, specify "netw_union_fname" and make sure the file exists')
        target_G = ab_union
    else:
        raise ValueError('Invalid value for parameter "target_netw": ' + target_netw)

    centr_atk_tactics = ['centrality_rank_from_bottom', 'centrality_rank_from_top', 'random_in_centrality_rank_range']

    if config.has_option(section, 'attacks'):
        node_cnt = config.getint(section, 'attacks')
    elif config.has_option(section, 'node_count'):
        node_cnt = config.getint(section, 'node_count')

    if method in centr_atk_tactics:
        ranked_nodes, bottom_skips, top_skips = \
            get_ranked_nodes(config, section, floader, netw_dir)

    if method == 'random':
        chosen_nodes = choose_random_nodes(target_G, node_cnt, seed)
    elif method == 'ith_node':
        node_rank = config.getint(section, 'node_rank')
        chosen_nodes = [pick_ith_node(target_G, node_rank)]
    elif method == 'targeted':
        target_nodes = config.get(section, 'target_nodes')
        # TODO: understand why this returns unicode even if it should not
        if (2, 7, 9) < sys.version_info < (3, 0, 0):
            target_nodes = str(target_nodes)
        chosen_nodes = [node for node in target_nodes.split()]  # split list on space
    elif method == 'centrality_rank_from_bottom':
        chosen_nodes = pick_nodes_by_rank_from_bottom(ranked_nodes, node_cnt, target_netw, I, bottom_skips)
    elif method == 'centrality_rank_from_top':
        chosen_nodes = pick_nodes_by_rank_from_top(ranked_nodes, node_cnt, target_netw, I, top_skips)
    elif method == 'random_in_centrality_rank_range':
        chosen_nodes = pick_random_nodes_in_rank_range(ranked_nodes, node_cnt, target_netw, I, bottom_skips,
                                                       top_skips, seed)
    elif method == 'most_inter_used':
        chosen_nodes = choose_most_inter_used_nodes(target_G, I, node_cnt, 'any')
    elif method == 'most_inter_used_distr_subs':
        chosen_nodes = choose_most_inter_used_nodes(target_G, I, node_cnt, 'distribution_substation')
    elif method == 'most_intra_used':
        chosen_nodes = choose_most_intra_used_nodes(target_G, node_cnt, 'any')
    elif method == 'most_intra_used_distr_subs':
        chosen_nodes = choose_most_intra_used_nodes(target_G, node_cnt, 'distribution_substation')
    elif method == 'most_intra_used_transm_subs':
        chosen_nodes = choose_most_intra_used_nodes(target_G, node_cnt, 'transmission_substation')
    elif method == 'most_intra_used_generators':
        chosen_nodes = choose_most_intra_used_nodes(target_G, node_cnt, 'generator')
    else:
        raise ValueError('Invalid value for parameter "attack_tactic": ' + method)

    return chosen_nodes


# this function will be called from another script, each time with a different configuration fpath
def run(conf_fpath, floader):
    global logger
    logger.info('conf_fpath = {}'.format(conf_fpath))

    conf_fpath = os.path.normpath(conf_fpath)
    if os.path.isabs(conf_fpath) is False:
        conf_fpath = os.path.abspath(conf_fpath)
    if not os.path.isfile(conf_fpath):
        raise ValueError('Invalid value for parameter "conf_fpath", no such file.\nPath: ' + conf_fpath)
    config = ConfigParser()
    config.read(conf_fpath)

    global time
    time = 0

    # read graphml files and instantiate network graphs

    seed = config.getint('run_opts', 'seed')

    if config.has_option('run_opts', 'save_death_cause'):
        save_death_cause = config.getboolean('run_opts', 'save_death_cause')
    else:
        save_death_cause = False

    # TODO: check if directory/files exist
    netw_dir = os.path.normpath(config.get('paths', 'netw_dir'))
    if os.path.isabs(netw_dir) is False:
        netw_dir = os.path.abspath(netw_dir)
    netw_a_fname = config.get('paths', 'netw_a_fname')
    netw_a_fpath_in = os.path.join(netw_dir, netw_a_fname)
    A = floader.fetch_graphml(netw_a_fpath_in, str)

    netw_b_fname = config.get('paths', 'netw_b_fname')
    netw_b_fpath_in = os.path.join(netw_dir, netw_b_fname)
    B = floader.fetch_graphml(netw_b_fpath_in, str)

    netw_inter_fname = config.get('paths', 'netw_inter_fname')
    netw_inter_fpath_in = os.path.join(netw_dir, netw_inter_fname)
    I = floader.fetch_graphml(netw_inter_fpath_in, str)

    # if the union graph is needed for this simulation, it's better to have it created in advance and just load it
    if config.has_option('paths', 'netw_union_fname'):
        netw_union_fname = config.get('paths', 'netw_union_fname')
        netw_union_fpath_in = os.path.join(netw_dir, netw_union_fname)
        ab_union = floader.fetch_graphml(netw_union_fpath_in, str)
    else:
        ab_union = None

    # read run options

    # nodes that must remain alive no matter what
    safe_nodes = []
    safe_nodes_a = set()
    safe_nodes_b = set()
    if config.has_section('safe_nodes_opts'):
        from_netw = config.get('safe_nodes_opts', 'from_netw')
        safe_sel_tactic = config.get('safe_nodes_opts', 'selection_tactic')
        safe_nodes = choose_nodes_by_config(config, 'safe_nodes_opts', from_netw, safe_sel_tactic, A, B, I, ab_union,
                                            floader, netw_dir, seed)

        for node in safe_nodes:
            node_netw = I.node[node]['network']
            if node_netw == A.graph['name']:
                # put the node name in a list, or each char becomes a separate set element
                safe_nodes_a.update([node])
            elif node_netw == B.graph['name']:
                safe_nodes_b.update([node])

    if len(safe_nodes) > 0 and config.has_option('run_opts', 'attacks'):
        logger.info('Nodes marked as safe will not be attacked, so if they get selected to be attacked, the number '
                    'of attacked nodes will be lower than specified!')

    attacked_netw = config.get('run_opts', 'attacked_netw')
    attack_tactic = config.get('run_opts', 'attack_tactic')
    attacked_nodes = choose_nodes_by_config(config, 'run_opts', attacked_netw, attack_tactic, A, B, I, ab_union,
                                            floader, netw_dir, seed)

    intra_support_type = config.get('run_opts', 'intra_support_type')
    if intra_support_type == 'cluster_size':
        min_cluster_size = config.getint('run_opts', 'min_cluster_size')
    inter_support_type = config.get('run_opts', 'inter_support_type')

    # read output paths

    results_dir = os.path.normpath(config.get('paths', 'results_dir'))
    if os.path.isabs(results_dir) is False:
        results_dir = os.path.abspath(results_dir)
    sf.ensure_dir_exists(results_dir)

    # run_stats is a file used to save step-by-step details about this run
    run_stats_fpath = ''
    run_stats_file = None
    run_stats = None
    if config.has_option('paths', 'run_stats_fname'):
        run_stats_fname = config.get('paths', 'run_stats_fname')
        run_stats_fpath = os.path.join(results_dir, run_stats_fname)

    # end_stats is a file used to save a single line (row) of statistics at the end of the simulation
    end_stats_fpath = ''
    if config.has_option('paths', 'end_stats_fpath'):
        end_stats_fpath = os.path.normpath(config.get('paths', 'end_stats_fpath'))
        if os.path.isabs(end_stats_fpath) is False:
            end_stats_fpath = os.path.abspath(end_stats_fpath)
        sf.ensure_dir_exists(os.path.dirname(end_stats_fpath))

    # ml_stats is similar to end_stats as it is used to write a line at the end of the simulation,
    # but it gathers statistics used for machine learning
    ml_stats_fpath = ''
    if config.has_option('paths', 'ml_stats_fpath'):
        ml_stats_fpath = os.path.normpath(config.get('paths', 'ml_stats_fpath'))
        if os.path.isabs(ml_stats_fpath) is False:
            ml_stats_fpath = os.path.abspath(ml_stats_fpath)

    # read information used to identify this simulation run
    sim_group = config.getint('misc', 'sim_group')
    instance = config.getint('misc', 'instance')

    batch_conf_fpath = ''
    if config.has_option('paths', 'batch_conf_fpath'):
        batch_conf_fpath = config.get('paths', 'batch_conf_fpath')

    if config.has_option('misc', 'run'):
        run_num = config.getint('misc', 'run')
    else:
        run_num = 0

    # statistics meant for machine learning are stored in this dictionary
    ml_stats = {'batch_conf_fpath': batch_conf_fpath, 'sim_group': sim_group, 'instance': instance,
                'run': run_num, 'seed': seed}

    # stability check
    unstable_nodes = set()
    if inter_support_type == 'node_interlink':
        unstable_nodes.update(find_nodes_without_inter_links(A, I))
    # elif inter_support_type == 'cluster_interlink':
    #     unstable_nodes.update(find_nodes_in_unsupported_clusters(A, I))
    elif inter_support_type == 'realistic':
        unstable_nodes.update(find_uncontrolled_pow_nodes(A, B, I))
    else:
        raise ValueError('Invalid value for parameter "inter_support_type": ' + inter_support_type)

    if intra_support_type == 'giant_component':
        unstable_nodes.update(find_nodes_not_in_giant_component(A))
    elif intra_support_type == 'cluster_size':
        unstable_nodes.update(find_nodes_in_smaller_clusters(A, min_cluster_size))
    elif intra_support_type == 'realistic':
        unstable_nodes.update(find_unpowered_substations(A))
    else:
        raise ValueError('Invalid value for parameter "intra_support_type": ' + intra_support_type)

    if inter_support_type == 'node_interlink':
        unstable_nodes.update(find_nodes_without_inter_links(B, I))
    # elif inter_support_type == 'cluster_interlink':
    #     unstable_nodes.update(find_nodes_in_unsupported_clusters(B, I))
    elif inter_support_type == 'realistic':
        unstable_nodes.update(find_nodes_without_inter_links(B, I))

    if intra_support_type == 'giant_component':
        unstable_nodes.update(find_nodes_not_in_giant_component(B))
    elif intra_support_type == 'cluster_size':
        unstable_nodes.update(find_nodes_in_smaller_clusters(B, min_cluster_size))
    # TODO: find out why we skipped / did not implement stability checks for this case
    # elif intra_support_type == 'realistic':
    #     unstable_nodes.update(list())

    # remove nodes that can't fail
    if len(safe_nodes_a) > 0 or len(safe_nodes_b) > 0:
        unstable_nodes -= safe_nodes_a.union(safe_nodes_b)

    if len(unstable_nodes) > 0:
        logger.debug('Time {}) {} nodes unstable before the initial attack: {}'.format(
            time, len(unstable_nodes), sorted(unstable_nodes, key=sf.natural_sort_key)))

    total_dead_a = 0
    total_dead_b = 0
    intra_sup_deaths_a = 0
    intra_sup_deaths_b = 0
    inter_sup_deaths_a = 0
    inter_sup_deaths_b = 0

    # only used if save_death_cause is True and inter_support_type == 'realistic'
    no_sup_ccs_deaths = 0
    no_sup_relays_deaths = 0
    no_com_path_deaths = 0

    base_node_cnt_a = A.number_of_nodes()
    base_node_cnt_b = B.number_of_nodes()

    # execute simulation of failure propagation
    try:
        if run_stats_fpath:
            run_stats_file = open(run_stats_fpath, 'wb')
            run_stats_header = ['time', 'dead']
            run_stats = csv.DictWriter(run_stats_file, run_stats_header, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            run_stats.writeheader()

        time += 1

        # perform initial attack

        attacked_nodes_a = []  # nodes in network A hit by the initial attack
        attacked_nodes_b = []  # nodes in network B hit by the initial attack
        dead_nodes_a = []
        dead_nodes_b = []
        for node in attacked_nodes:
            node_netw = I.node[node]['network']
            if node_netw == A.graph['name']:
                attacked_nodes_a.append(node)
            elif node_netw == B.graph['name']:
                attacked_nodes_b.append(node)
            else:
                raise RuntimeError('Node {} network "{}" marked in inter graph is neither A nor B'.format(
                    node, node_netw))

        if ml_stats_fpath:  # if this string is not empty
            ml_stats.update({'#atkd': len(attacked_nodes_a) + len(attacked_nodes_b), '#atkd_a': len(attacked_nodes_a),
                             '#atkd_b': len(attacked_nodes_b)})
            ml_stats.update({'atkd_nodes_a': attacked_nodes_a, 'atkd_nodes_b': attacked_nodes_b})
            ml_stats.update(
                calc_atk_centr_stats(A.graph['name'], B.graph['name'], I.graph['name'], ab_union.graph['name'],
                                     attacked_nodes_a, attacked_nodes_b, floader, netw_dir))

            result_key_by_role = {'generator': 'p_atkd_gen', 'transmission_substation': 'p_atkd_ts',
                                  'distribution_substation': 'p_atkd_ds'}
            ml_stats.update(calc_atkd_percent_by_role(A, attacked_nodes_a, result_key_by_role))

            result_key_by_role = {'relay': 'p_atkd_rel', 'controller': 'p_atkd_cc'}
            ml_stats.update(calc_atkd_percent_by_role(B, attacked_nodes_b, result_key_by_role))

        total_dead_a = len(attacked_nodes_a)
        if total_dead_a > 0:
            dead_nodes_a.extend(attacked_nodes_a)
            A.remove_nodes_from(attacked_nodes_a)
            logger.info('Time {}) {} nodes of network {} failed because of initial attack: {}'.format(
                time, total_dead_a, A.graph['name'], sorted(attacked_nodes_a, key=sf.natural_sort_key)))

        total_dead_b = len(attacked_nodes_b)
        if total_dead_b > 0:
            dead_nodes_b.extend(attacked_nodes_b)
            B.remove_nodes_from(attacked_nodes_b)
            logger.info('Time {}) {} nodes of network {} failed because of initial attack: {}'.format(
                time, total_dead_b, B.graph['name'], sorted(attacked_nodes_b, key=sf.natural_sort_key)))

        if total_dead_a == total_dead_b == 0:
            logger.info('Time {}) No nodes were attacked'.format(time))

        I.remove_nodes_from(attacked_nodes)

        # save_state(time, A, B, I, results_dir)
        if run_stats is not None:
            run_stats.writerow({'time': time, 'dead': attacked_nodes})

        updated = True
        time += 1

        # phase checks

        while updated is True:
            updated = False

            # inter checks for network A
            if inter_support_type == 'node_interlink':
                unsupported_nodes_a = find_nodes_without_inter_links(A, I)
            # elif inter_support_type == 'cluster_interlink':
            #     unsupported_nodes_a = find_nodes_in_unsupported_clusters(A, I)
            elif inter_support_type == 'realistic':
                unsupported_nodes_a = find_uncontrolled_pow_nodes(A, B, I, save_death_cause)

            if save_death_cause is False:
                unsupported_nodes_a = remove_list_items(unsupported_nodes_a, safe_nodes_a)
            elif inter_support_type == 'realistic':  # save_death_cause is True
                remove_items_from_lists_in_dict(unsupported_nodes_a, safe_nodes_a)
                no_sup_ccs_deaths += len(unsupported_nodes_a['no_sup_ccs'])
                no_sup_relays_deaths += len(unsupported_nodes_a['no_sup_relays'])
                no_com_path_deaths += len(unsupported_nodes_a['no_com_path'])
                temp_list = []
                for node_list in unsupported_nodes_a.values():  # convert dictionary of lists into a simple list
                    temp_list.extend(node_list)
                unsupported_nodes_a = temp_list

            failed_cnt_a = len(unsupported_nodes_a)
            if failed_cnt_a > 0:
                logger.info('Time {}) {} nodes of network {} failed for lack of inter support: {}'.format(
                    time, failed_cnt_a, A.graph['name'], sorted(unsupported_nodes_a, key=sf.natural_sort_key)))
                total_dead_a += failed_cnt_a
                inter_sup_deaths_a += failed_cnt_a
                dead_nodes_a.extend(unsupported_nodes_a)
                A.remove_nodes_from(unsupported_nodes_a)
                I.remove_nodes_from(unsupported_nodes_a)
                updated = True
                # save_state(time, A, B, I, results_dir)
                if run_stats is not None:
                    run_stats.writerow({'time': time, 'dead': unsupported_nodes_a})
            time += 1

            # intra checks for network A
            if intra_support_type == 'giant_component':
                unsupported_nodes_a = find_nodes_not_in_giant_component(A)
            elif intra_support_type == 'cluster_size':
                unsupported_nodes_a = find_nodes_in_smaller_clusters(A, min_cluster_size)
            elif intra_support_type == 'realistic':
                unsupported_nodes_a = find_unpowered_substations(A)

            unsupported_nodes_a = remove_list_items(unsupported_nodes_a, safe_nodes_a)
            failed_cnt_a = len(unsupported_nodes_a)
            if failed_cnt_a > 0:
                logger.info('Time {}) {} nodes of network {} failed for lack of intra support: {}'.format(
                    time, failed_cnt_a, A.graph['name'], sorted(unsupported_nodes_a, key=sf.natural_sort_key)))
                total_dead_a += failed_cnt_a
                intra_sup_deaths_a += failed_cnt_a
                dead_nodes_a.extend(unsupported_nodes_a)
                A.remove_nodes_from(unsupported_nodes_a)
                I.remove_nodes_from(unsupported_nodes_a)
                updated = True
                # save_state(time, A, B, I, results_dir)
                if run_stats is not None:
                    run_stats.writerow({'time': time, 'dead': unsupported_nodes_a})
            time += 1

            # inter checks for network B
            if inter_support_type == 'node_interlink':
                unsupported_nodes_b = find_nodes_without_inter_links(B, I)
            # elif inter_support_type == 'cluster_interlink':
            #     unsupported_nodes_b = find_nodes_in_unsupported_clusters(B, I)
            elif inter_support_type == 'realistic':
                unsupported_nodes_b = find_nodes_without_inter_links(B, I)

            unsupported_nodes_b = remove_list_items(unsupported_nodes_b, safe_nodes_b)
            failed_cnt_b = len(unsupported_nodes_b)
            if failed_cnt_b > 0:
                logger.info('Time {}) {} nodes of network {} failed for lack of inter support: {}'.format(
                    time, failed_cnt_b, B.graph['name'], sorted(unsupported_nodes_b, key=sf.natural_sort_key)))
                total_dead_b += failed_cnt_b
                inter_sup_deaths_b += failed_cnt_b
                dead_nodes_b.extend(unsupported_nodes_b)
                B.remove_nodes_from(unsupported_nodes_b)
                I.remove_nodes_from(unsupported_nodes_b)
                updated = True
                # save_state(time, A, B, I, results_dir)
                if run_stats is not None:
                    run_stats.writerow({'time': time, 'dead': unsupported_nodes_b})
            time += 1

            # intra checks for network B
            if intra_support_type == 'giant_component':
                unsupported_nodes_b = find_nodes_not_in_giant_component(B)
            elif intra_support_type == 'cluster_size':
                unsupported_nodes_b = find_nodes_in_smaller_clusters(B, min_cluster_size)
            elif intra_support_type == 'realistic':
                # in the realistic model, telecom nodes can survive without intra support, they just cant't communicate
                unsupported_nodes_b = []

            unsupported_nodes_b = remove_list_items(unsupported_nodes_b, safe_nodes_b)
            failed_cnt_b = len(unsupported_nodes_b)
            if failed_cnt_b > 0:
                logger.info('Time {}) {} nodes of network {} failed for lack of intra support: {}'.format(
                    time, failed_cnt_b, B.graph['name'], sorted(unsupported_nodes_b, key=sf.natural_sort_key)))
                total_dead_b += failed_cnt_b
                intra_sup_deaths_b += failed_cnt_b
                dead_nodes_b.extend(unsupported_nodes_b)
                B.remove_nodes_from(unsupported_nodes_b)
                I.remove_nodes_from(unsupported_nodes_b)
                updated = True
                # save_state(time, A, B, I, results_dir)
                if run_stats is not None:
                    run_stats.writerow({'time': time, 'dead': unsupported_nodes_b})
            time += 1
    except:
        if run_stats_file is not None:
            run_stats_file.close()
        raise

    # save_state('final', A, B, I, results_dir)

    # write statistics about the final result

    if end_stats_fpath:  # if this string is not empty
        end_stats_file_existed = os.path.isfile(end_stats_fpath)
        with open(end_stats_fpath, 'ab') as end_stats_file:
            end_stats_header = ['batch_conf_fpath', 'sim_group', 'instance', 'run', '#dead', '#dead_a', '#dead_b']

            if save_death_cause is True:
                end_stats_header.extend(['no_intra_sup_a', 'no_inter_sup_a',
                                         'no_intra_sup_b', 'no_inter_sup_b'])
                if inter_support_type == 'realistic':
                    end_stats_header.extend(['no_sup_ccs', 'no_sup_relays', 'no_com_path'])

            end_stats_writer = csv.DictWriter(end_stats_file, end_stats_header, delimiter='\t',
                                              quoting=csv.QUOTE_MINIMAL)
            if end_stats_file_existed is False:
                end_stats_writer.writeheader()

            end_stats_row = {'batch_conf_fpath': batch_conf_fpath, 'sim_group': sim_group, 'instance': instance,
                             'run': run_num, '#dead': total_dead_a + total_dead_b,
                             '#dead_a': total_dead_a, '#dead_b': total_dead_b}

            if save_death_cause is True:
                end_stats_row.update({'no_intra_sup_a': intra_sup_deaths_a, 'no_inter_sup_a': inter_sup_deaths_a,
                                      'no_intra_sup_b': intra_sup_deaths_b, 'no_inter_sup_b': inter_sup_deaths_b})
                if inter_support_type == 'realistic':
                    end_stats_row.update({'no_sup_ccs': no_sup_ccs_deaths, 'no_sup_relays': no_sup_relays_deaths,
                                          'no_com_path': no_com_path_deaths})
            end_stats_writer.writerow(end_stats_row)

    if ml_stats_fpath:  # if this string is not empty

        # percentages of dead nodes over the initial number of nodes in the graph
        ml_stats['p_dead_a'] = sf.percent_of_part(total_dead_a, base_node_cnt_a)
        ml_stats['p_dead_b'] = sf.percent_of_part(total_dead_b, base_node_cnt_b)
        ml_stats['p_dead'] = sf.percent_of_part(total_dead_a + total_dead_b, base_node_cnt_a + base_node_cnt_b)
        ml_stats['dead_nodes_a'] = dead_nodes_a
        ml_stats['dead_nodes_b'] = dead_nodes_b
        ml_stats['dead_count_a'] = len(dead_nodes_a)
        ml_stats['dead_count_b'] = len(dead_nodes_b)
        ml_stats['dead_count'] = len(dead_nodes_a) + len(dead_nodes_b)

        if config.has_section('safe_nodes_opts'):
            ml_stats['safe_count'] = len(safe_nodes)

        ml_stats_file_existed = os.path.isfile(ml_stats_fpath)
        with open(ml_stats_fpath, 'ab') as ml_stats_file:

            # sort statistics columns by name so they can be more found easily in the output file
            ml_stats_header = sorted(ml_stats.keys(), key=sf.natural_sort_key)

            ml_stats_writer = csv.DictWriter(ml_stats_file, ml_stats_header, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            if ml_stats_file_existed is False:
                ml_stats_writer.writeheader()

            ml_stats_writer.writerow(ml_stats)
