__author__ = 'Agostino Sturaro'

import os
import logging
import networkx as nx
import random
import csv
from collections import OrderedDict
import shared_functions as sf

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0


# global variables
logger = None
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


def pick_nodes_by_betw_centr(G, node_cnt, seed=None):
    nodes_with_centr = nx.betweenness_centrality(G, seed=seed)
    # sort the dictionary by value first and then by key (so we have a deterministic sorting)
    nodes_with_rank = OrderedDict(sorted(nodes_with_centr.items(), key=lambda (k, v): (v, k)))
    chosen_nodes = list()
    # choose the first nodes from the ordered dictionary
    for i, node in enumerate(nodes_with_rank):
        if i >= node_cnt:
            break
        chosen_nodes.append(node)
    return chosen_nodes, nodes_with_centr


def pick_nodes_by_betw_centr_rank(G, node_cnt, min_rank, seed=None):
    nodes_with_centr = nx.betweenness_centrality(G, seed=seed)
    # sort the dictionary by value first and then by key (so we have a deterministic sorting)
    # we get a dictionary of node: centrality
    nodes_with_rank = OrderedDict(sorted(nodes_with_centr.items(), key=lambda (k, v): (v, k)))
    # create a list of nodes rank by their centrality, we get a list
    # [node with lowest centrality, ..., node with highest centrality]
    ranked_nodes = list(nodes_with_rank.keys())
    chosen_nodes = list()
    for i in range(min_rank, min_rank + node_cnt):
        node = ranked_nodes[i]
        chosen_nodes.append(node)
    return chosen_nodes


def pick_nodes_by_closeness_centr_rank(G, node_cnt, min_rank):
    nodes_with_centr = nx.closeness_centrality(G)
    # sort the dictionary by value first and then by key (so we have a deterministic sorting)
    # we get a dictionary of node: centrality
    nodes_with_rank = OrderedDict(sorted(nodes_with_centr.items(), key=lambda (k, v): (v, k)))
    # create a list of nodes rank by their centrality, we get a list
    # [node with lowest centrality, ..., node with highest centrality]
    ranked_nodes = list(nodes_with_rank.keys())
    chosen_nodes = list()
    for i in range(min_rank, min_rank + node_cnt):
        node = ranked_nodes[i]
        chosen_nodes.append(node)
    return chosen_nodes


# nodes_with_centr is a dictionary {'node': centrality measure}
# node_cnt is the number of nodes to pick
# min_rank is the rank of the first node to pick (they get sorted by centrality measure and id)
def pick_nodes_by_centrality_rank(nodes_with_centr, node_cnt, min_rank):
    # sort the dictionary by value first and then by key (so we have a deterministic sorting)
    # we get a dictionary of node: centrality
    nodes_with_rank = OrderedDict(sorted(nodes_with_centr.items(), key=lambda (k, v): (v, k)))
    # create a list of nodes rank by their centrality, we get a list
    # [node with lowest centrality, ..., node with highest centrality]
    ranked_nodes = list(nodes_with_rank.keys())
    chosen_nodes = list()
    for i in range(min_rank, min_rank + node_cnt):
        node = ranked_nodes[i]
        chosen_nodes.append(node)
    return chosen_nodes


def choose_most_inter_used_nodes(G, I, node_cnt, role):
    # select distribution substations from G
    # get the in-degree of each distribution substation from I
    # make that a list of tuples
    nodes_with_rank = list()
    for node in G.nodes():
        node_role = G.node[node]['role']
        if node_role == role:
            if nx.is_directed(I):
                rank = I.in_degree(node)
            else:
                rank = I.degree(node)
            nodes_with_rank.append((rank, node))

    # sort the data structure by degree and node id
    nodes_with_rank.sort()

    # pick the first node_cnt nodes from the data structure
    # put them in a list and return it
    chosen_nodes = list()
    for i in range(0, node_cnt):
        chosen_nodes.append(nodes_with_rank.pop()[1])
    return chosen_nodes


def choose_most_intra_used_nodes(G, node_cnt, role):
    # select nodes from G, get their degree
    # make that a list of tuples
    nodes_with_rank = list()
    for node in G.nodes():
        node_role = G.node[node]['role']
        if node_role == role:
            rank = G.degree(node)
            nodes_with_rank.append((rank, node))

    # sort the data structure by degree and node id
    nodes_with_rank.sort()

    # pick the first node_cnt nodes from the data structure
    # put them in a list and return it
    chosen_nodes = list()
    for i in range(0, node_cnt):
        chosen_nodes.append(nodes_with_rank.pop()[1])
    return chosen_nodes


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


# this function will be called from another script, each time with a different configuration fpath
def run(conf_fpath):
    global logger
    logger = logging.getLogger(__name__)
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

    seed = config.get('run_opts', 'seed')

    if config.has_option('run_opts', 'save_death_cause'):
        save_death_cause = config.getboolean('run_opts', 'save_death_cause')
    else:
        save_death_cause = False

    netw_dir = os.path.normpath(config.get('paths', 'netw_dir'))
    if os.path.isabs(netw_dir) is False:
        netw_dir = os.path.abspath(netw_dir)
    netw_a_fname = config.get('paths', 'netw_a_fname')
    netw_a_fpath_in = os.path.join(netw_dir, netw_a_fname)
    A = nx.read_graphml(netw_a_fpath_in, node_type=str)

    netw_b_fname = config.get('paths', 'netw_b_fname')
    netw_b_fpath_in = os.path.join(netw_dir, netw_b_fname)
    B = nx.read_graphml(netw_b_fpath_in, node_type=str)

    netw_inter_fname = config.get('paths', 'netw_inter_fname')
    netw_inter_fpath_in = os.path.join(netw_dir, netw_inter_fname)
    I = nx.read_graphml(netw_inter_fpath_in, node_type=str)

    # read run options

    attacked_netw = config.get('run_opts', 'attacked_netw')
    attack_tactic = config.get('run_opts', 'attack_tactic')
    if attack_tactic != 'targeted':
        attack_cnt = config.getint('run_opts', 'attacks')
    intra_support_type = config.get('run_opts', 'intra_support_type')
    if intra_support_type == 'cluster_size':
        min_cluster_size = config.getint('run_opts', 'min_cluster_size')
    inter_support_type = config.get('run_opts', 'inter_support_type')

    # read output paths

    # run_stats is a file used to save step-by-step details about this run
    results_dir = os.path.normpath(config.get('paths', 'results_dir'))
    if os.path.isabs(results_dir) is False:
        results_dir = os.path.abspath(results_dir)
    run_stats_fname = config.get('paths', 'run_stats_fname')
    run_stats_fpath = os.path.join(results_dir, run_stats_fname)

    # end_stats is a file used to save a single line (row) of statistics at the end of the simulation
    end_stats_fpath = os.path.normpath(config.get('paths', 'end_stats_fpath'))
    if os.path.isabs(end_stats_fpath) is False:
        end_stats_fpath = os.path.abspath(end_stats_fpath)

    # ml_stats is similar to end_stats as it is used to write a line at the end of the simulation,
    # but it gathers statistics used for machine learning
    ml_stats_fpath = ''
    if config.has_option('paths', 'ml_stats_fpath'):
        ml_stats_fpath = os.path.normpath(config.get('paths', 'ml_stats_fpath'))
        if os.path.isabs(ml_stats_fpath) is False:
            ml_stats_fpath = os.path.abspath(ml_stats_fpath)

    # ensure output directories exist and are empty
    sf.ensure_dir_exists(results_dir)
    sf.ensure_dir_exists(os.path.dirname(end_stats_fpath))

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
    # elif intra_support_type == 'realistic':
    #     unstable_nodes.update(list())

    if len(unstable_nodes) > 0:
        logger.debug('Time {}) {} nodes unstable before the initial attack: {}'.format(
            time, len(unstable_nodes), sorted(unstable_nodes, key=sf.natural_sort_key)))

    total_dead_a = 0
    total_dead_b = 0
    intra_sup_deaths_a = 0
    intra_sup_deaths_b = 0
    inter_sup_deaths_a = 0
    inter_sup_deaths_b = 0

    if save_death_cause is True and inter_support_type == 'realistic':
        no_sup_ccs_deaths = 0
        no_sup_relays_deaths = 0
        no_com_path_deaths = 0

    # execute simulation of failure propagation
    with open(run_stats_fpath, 'wb') as run_stats_file:

        run_stats = csv.DictWriter(run_stats_file, ['time', 'dead'], delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        run_stats.writeheader()
        time += 1

        # perform initial attack
        if attacked_netw == A.graph['name']:
            attacked_G = A
        elif attacked_netw == B.graph['name']:
            attacked_G = B
        elif attacked_netw == "both":
            attacked_G = nx.compose(nx.compose(A, I), B, "both")
        else:
            raise ValueError('Invalid value for parameter "attacked_netw": ' + attacked_netw)

        attacks_for_A_only = ['most_inter_used_distr_subs', 'most_intra_used_distr_subs', 'most_intra_used_transm_subs',
                              'most_intra_used_generators']
        if attack_tactic in attacks_for_A_only and attacked_netw != A.graph['name']:
            raise ValueError('Attack {} can only be used on the power network A'.format(attack_tactic))

        betw_c_by_node = None
        clos_c_by_node = None
        indeg_c_by_node = None

        if attack_tactic == 'random':
            attacked_nodes = choose_random_nodes(attacked_G, attack_cnt, seed)
        elif attack_tactic == 'targeted':
            target_nodes = config.get('run_opts', 'target_nodes')
            attacked_nodes = [node for node in target_nodes.split()]  # split list on space
        # elif attack_tactic == 'betweenness_centrality':
        #     attacked_nodes, betw_c_by_node = pick_nodes_by_betw_centr(attacked_G, attack_cnt, seed)
        #     clos_c_by_node = nx.closeness_centrality(attacked_G)
        elif attack_tactic == 'betweenness_centrality_rank':
            betw_c_by_node = nx.betweenness_centrality(attacked_G, seed=seed)
            min_rank = config.getint('run_opts', 'min_rank')
            attacked_nodes = pick_nodes_by_centrality_rank(betw_c_by_node, attack_cnt, min_rank)
        elif attack_tactic == 'closeness_centrality_rank':
            clos_c_by_node = nx.closeness_centrality(attacked_G)
            min_rank = config.getint('run_opts', 'min_rank')
            attacked_nodes = pick_nodes_by_centrality_rank(clos_c_by_node, attack_cnt, min_rank)
        elif attack_tactic == 'most_inter_used_distr_subs':
            attacked_nodes = choose_most_inter_used_nodes(attacked_G, I, attack_cnt, 'distribution_substation')
        elif attack_tactic == 'most_intra_used_distr_subs':
            attacked_nodes = choose_most_intra_used_nodes(attacked_G, attack_cnt, 'distribution_substation')
        elif attack_tactic == 'most_intra_used_transm_subs':
            attacked_nodes = choose_most_intra_used_nodes(attacked_G, attack_cnt, 'transmission_substation')
        elif attack_tactic == 'most_intra_used_generators':
            attacked_nodes = choose_most_intra_used_nodes(attacked_G, attack_cnt, 'generator')
        else:
            raise ValueError('Invalid value for parameter "attack_tactic": ' + attack_tactic)

        # calculate additional centrality measures only if we need to fill ML stats
        if ml_stats_fpath:
            if betw_c_by_node is None:
                betw_c_by_node = nx.betweenness_centrality(attacked_G, seed=seed)
            if clos_c_by_node is None:
                clos_c_by_node = nx.closeness_centrality(attacked_G)
            if indeg_c_by_node is None:
                indeg_c_by_node = nx.in_degree_centrality(attacked_G)

            # small hack, we have only 1 node attacked
            atkd_clos_c = clos_c_by_node(attacked_nodes[0])
            atkd_betw_c = betw_c_by_node(attacked_nodes[0])
            atkd_indeg_c = indeg_c_by_node(attacked_nodes[0])
            atkd_role = attacked_nodes[0][0]  # the first character in the name of the node is its role

        attacked_nodes_a = set(attacked_nodes).intersection(A.nodes())
        attacked_nodes_b = set(attacked_nodes).intersection(B.nodes())

        total_dead_a = len(attacked_nodes_a)
        if total_dead_a > 0:
            A.remove_nodes_from(attacked_nodes_a)
            logger.info('Time {}) {} nodes of network {} failed because of initial attack: {}'.format(
                time, total_dead_a, A.graph['name'], sorted(attacked_nodes_a, key=sf.natural_sort_key)))

        total_dead_b = len(attacked_nodes_b)
        if total_dead_b > 0:
            B.remove_nodes_from(attacked_nodes_b)
            logger.info('Time {}) {} nodes of network {} failed because of initial attack: {}'.format(
                time, total_dead_b, B.graph['name'], sorted(attacked_nodes_b, key=sf.natural_sort_key)))

        if total_dead_a == total_dead_b == 0:
            logger.info('Time {}) No nodes were attacked'.format(time))

        I.remove_nodes_from(attacked_nodes)

        # save_state(time, A, B, I, results_dir)
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

            if save_death_cause is True and inter_support_type == 'realistic':
                no_sup_ccs_deaths += len(unsupported_nodes_a['no_sup_ccs'])
                no_sup_relays_deaths += len(unsupported_nodes_a['no_sup_relays'])
                no_com_path_deaths += len(unsupported_nodes_a['no_com_path'])
                temp_list = list()

                # convert dictionary of lists to simple list
                for node_list in unsupported_nodes_a.values():
                    temp_list.extend(node_list)
                unsupported_nodes_a = temp_list

            failed_cnt_a = len(unsupported_nodes_a)
            if failed_cnt_a > 0:
                logger.info('Time {}) {} nodes of network {} failed for lack of inter support: {}'.format(
                    time, failed_cnt_a, A.graph['name'], sorted(unsupported_nodes_a, key=sf.natural_sort_key)))
                total_dead_a += failed_cnt_a
                inter_sup_deaths_a += failed_cnt_a
                A.remove_nodes_from(unsupported_nodes_a)
                I.remove_nodes_from(unsupported_nodes_a)
                updated = True
                # save_state(time, A, B, I, results_dir)
                run_stats.writerow({'time': time, 'dead': unsupported_nodes_a})
            time += 1

            # intra checks for network A
            if intra_support_type == 'giant_component':
                unsupported_nodes_a = find_nodes_not_in_giant_component(A)
            elif intra_support_type == 'cluster_size':
                unsupported_nodes_a = find_nodes_in_smaller_clusters(A, min_cluster_size)
            elif intra_support_type == 'realistic':
                unsupported_nodes_a = find_unpowered_substations(A)
            failed_cnt_a = len(unsupported_nodes_a)
            if failed_cnt_a > 0:
                logger.info('Time {}) {} nodes of network {} failed for lack of intra support: {}'.format(
                    time, failed_cnt_a, A.graph['name'], sorted(unsupported_nodes_a, key=sf.natural_sort_key)))
                total_dead_a += failed_cnt_a
                intra_sup_deaths_a += failed_cnt_a
                A.remove_nodes_from(unsupported_nodes_a)
                I.remove_nodes_from(unsupported_nodes_a)
                updated = True
                # save_state(time, A, B, I, results_dir)
                run_stats.writerow({'time': time, 'dead': unsupported_nodes_a})
            time += 1

            # inter checks for network B
            if inter_support_type == 'node_interlink':
                unsupported_nodes_b = find_nodes_without_inter_links(B, I)
            # elif inter_support_type == 'cluster_interlink':
            #     unsupported_nodes_b = find_nodes_in_unsupported_clusters(B, I)
            elif inter_support_type == 'realistic':
                unsupported_nodes_b = find_nodes_without_inter_links(B, I)
            failed_cnt_b = len(unsupported_nodes_b)
            if failed_cnt_b > 0:
                logger.info('Time {}) {} nodes of network {} failed for lack of inter support: {}'.format(
                    time, failed_cnt_b, B.graph['name'], sorted(unsupported_nodes_b, key=sf.natural_sort_key)))
                total_dead_b += failed_cnt_b
                inter_sup_deaths_b += failed_cnt_b
                B.remove_nodes_from(unsupported_nodes_b)
                I.remove_nodes_from(unsupported_nodes_b)
                updated = True
                # save_state(time, A, B, I, results_dir)
                run_stats.writerow({'time': time, 'dead': unsupported_nodes_b})
            time += 1

            # intra checks for network B
            if intra_support_type == 'giant_component':
                unsupported_nodes_b = find_nodes_not_in_giant_component(B)
            elif intra_support_type == 'cluster_size':
                unsupported_nodes_b = find_nodes_in_smaller_clusters(B, min_cluster_size)
            elif intra_support_type == 'realistic':
                unsupported_nodes_b = list()
            failed_cnt_b = len(unsupported_nodes_b)
            if failed_cnt_b > 0:
                logger.info('Time {}) {} nodes of network {} failed for lack of intra support: {}'.format(
                    time, failed_cnt_b, B.graph['name'], sorted(unsupported_nodes_b, key=sf.natural_sort_key)))
                total_dead_b += failed_cnt_b
                intra_sup_deaths_b += failed_cnt_b
                B.remove_nodes_from(unsupported_nodes_b)
                I.remove_nodes_from(unsupported_nodes_b)
                updated = True
                # save_state(time, A, B, I, results_dir)
                run_stats.writerow({'time': time, 'dead': unsupported_nodes_b})
            time += 1

    # save_state('final', A, B, I, results_dir)

    # write statistics about the final result
    if os.path.isfile(end_stats_fpath) is False:
        write_header = True  # if we are going to create a new file, then add the header
    else:
        write_header = False
    with open(end_stats_fpath, 'ab') as end_stats_file:
        end_stats_header = ['total_dead']
        if save_death_cause is True:
            end_stats_header.extend(['no_intra_sup_a', 'no_inter_sup_a',
                                     'no_intra_sup_b', 'no_inter_sup_b'])
            if inter_support_type == 'realistic':
                end_stats_header.extend(['no_sup_ccs', 'no_sup_relays', 'no_com_path'])
        end_stats = csv.DictWriter(end_stats_file, end_stats_header, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        if write_header is True:
            end_stats.writeheader()
        end_stats_row = {'total_dead': total_dead_a + total_dead_b}
        if save_death_cause is True:
            end_stats_row.update({'no_intra_sup_a': intra_sup_deaths_a, 'no_inter_sup_a': inter_sup_deaths_a,
                                  'no_intra_sup_b': intra_sup_deaths_b, 'no_inter_sup_b': inter_sup_deaths_b})
            if inter_support_type == 'realistic':
                end_stats_row.update({'no_sup_ccs': no_sup_ccs_deaths, 'no_sup_relays': no_sup_relays_deaths,
                                      'no_com_path': no_com_path_deaths})
        end_stats.writerow(end_stats_row)

    if ml_stats_fpath:  # if this string is not empty
        if os.path.isfile(end_stats_fpath) is False:
            write_header = True  # if we are going to create a new file, then add the header
        else:
            write_header = False
        with open(ml_stats_fpath, 'ab') as ml_stats_file:
            # later we can add features about the whole network, like the median centrality
            # ml_stats_header = ['atkd_G', 'atkd_T', 'atkd_D', 'atkd_R', 'atkd_C', 'betw_c', ...]

            ml_stats_header = ['atkd_role', 'betw_c', 'clos_c', 'indeg_c', 'tot_dead']
            ml_stats = csv.DictWriter(ml_stats_file, ml_stats_header, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            if write_header is True:
                ml_stats.writeheader()

            ml_stats_row = {'atkd_role': atkd_role, 'betw_c': atkd_betw_c, 'clos_c': atkd_clos_c,
                            'indeg_c': atkd_indeg_c, 'tot_dead': total_dead_a + total_dead_b}
            ml_stats.writerow(ml_stats_row)
