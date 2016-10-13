import os
import logging
import networkx as nx
import random
import csv
import sys
import shared_functions as sf

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0

__author__ = 'Agostino Sturaro'

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


def pick_ith_node(G, node_rank):
    nodes = G.nodes()
    nodes.sort()
    return nodes[node_rank]


# ranked_nodes is a list of nodes, sorted in a deterministic way
# [node with lowest rank, ..., node with highest rank]
# node_cnt is the number of nodes to pick
# min_rank is the rank of the first node to pick
# TODO: throw away
def pick_nodes_by_rank(ranked_nodes, node_cnt, min_rank):
    chosen_nodes = list()
    for i in range(min_rank, min_rank + node_cnt):
        node = ranked_nodes[i]
        chosen_nodes.append(node)
    return chosen_nodes


def pick_nodes_by_role(G, role, only_nodes_in=None):
    chosen_nodes = list()
    if only_nodes_in is None:
        for node in G.nodes():
            node_role = G.node[node]['role']
            if node_role == role:
                chosen_nodes.append(node)
    else:
        for node in only_nodes_in:
            node_role = G.node[node]['role']
            if node_role == role:
                chosen_nodes.append(node)
    return chosen_nodes


# rank_node_pairs is a list of tuples (rank, node), nodes with the same rank get ordered by id and then can be shuffled
def choose_nodes_by_rank(rank_node_pairs, node_cnt, secondary_sort, seed=None):
    if len(rank_node_pairs) < node_cnt:
        raise RuntimeError("Insufficient nodes to select")

    chosen_nodes = list()

    # sort the data structure by rank and node id, so have a deterministic order
    sorted_pairs = sorted(rank_node_pairs)

    if secondary_sort == 'random_shuffle':
        chosen_nodes = list()
        for i in range(0, node_cnt):
            chosen_nodes.append(sorted_pairs.pop()[1])
    else:
        # shuffle nodes with the same degree
        my_random = random.Random(seed)
        current_rank = sorted_pairs[0];
        nodes_by_rank = {};

        for i in range(0, len(sorted_pairs)):
            rank, node = sorted_pairs[i]
            if rank != current_rank:
                nodes_by_rank[rank] = []
                current_rank = rank
            nodes_by_rank[rank].append(node)

        ranks = sorted(nodes_by_rank.keys(), reverse=True)
        for rank in ranks:
            nodes = nodes_by_rank[rank]
            if len(nodes_by_rank[rank]) > 1:
                my_random.shuffle(nodes)
            for node in nodes:
                if len(chosen_nodes) >= node_cnt:
                    break
                chosen_nodes.append(node)

            if len(chosen_nodes) >= node_cnt:
                break

    return chosen_nodes


def choose_most_inter_used_nodes(G, I, node_cnt, role, secondary_sort, seed=None):
    # select nodes with the specified role from graph G and create a list of tuples with their in-degree in graph I
    rank_node_pairs = list()
    for node in G.nodes():
        node_role = G.node[node]['role']
        if node_role == role:
            if nx.is_directed(I):
                rank = I.in_degree(node)
            else:
                rank = I.degree(node)
            rank_node_pairs.append((rank, node))

    return choose_nodes_by_rank(rank_node_pairs, node_cnt, secondary_sort, seed)


def choose_most_intra_used_nodes(G, node_cnt, role, secondary_sort, seed=None):
    # select nodes from G, get their degree
    # make that a list of tuples
    rank_node_pairs = list()
    for node in G.nodes():
        node_role = G.node[node]['role']
        if node_role == role:
            rank = G.degree(node)
            rank_node_pairs.append((rank, node))

    return choose_nodes_by_rank(rank_node_pairs, node_cnt, secondary_sort, seed)


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


# TODO: test this function
def calc_stats_on_centrality(attacked_nodes, full_centrality_name, short_centrality_name, file_loader,
                             centrality_file_path):
    # load file with precalculated centrality metrics
    centrality_info = file_loader.fetch_json(centrality_file_path)
    centr_by_node = centrality_info[full_centrality_name]

    # how many nodes are in each quintile
    node_cnt_by_quintile = [full_centrality_name + '_node_count_by_quintile']

    # count how many nodes were attacked in each quintile
    quintiles = centrality_info[full_centrality_name + '_quintiles']
    atkd_cnt_by_quintile = [0] * 5
    for node in attacked_nodes:
        node_quintile = 0
        node_centr = centr_by_node[node]
        for quintile in quintiles:
            if node_centr > quintile:
                node_quintile += 1
            else:
                break
        atkd_cnt_by_quintile[node_quintile] += 1

    # calc what percentage of the nodes in each quintile was attacked
    atkd_perc_of_quintile = [0.0] * 5
    for i in range(0, 5):
        atkd_perc_of_quintile[i] = sf.percent_of_part(atkd_cnt_by_quintile[i], node_cnt_by_quintile[i])

    # dictionary used to index statistics with shorter names
    centr_stats = {}

    # sum the centrality of the attacked nodes
    centr_sum = 0.0
    for node in attacked_nodes:
        centr_sum += centr_by_node[node]
    stat_name = short_centrality_name + '_sum'
    centr_stats[stat_name] = centr_sum

    for i, val in enumerate(atkd_perc_of_quintile):
        stat_name = short_centrality_name + '_q_' + str(i + 1)  # q_1 is the first quantile
        centr_stats[stat_name] = val

    return centr_stats


# TODO: make it all uniform
def calc_centrality_fraction(attacked_nodes, full_centrality_name, file_loader, centrality_file_path):
    if full_centrality_name == 'degree_centrality':
        raise RuntimeError('Call calc_degree_centr_fraction instead')
    elif full_centrality_name == 'degree_centrality':  # redundant, already saved!
        raise RuntimeError('Call calc_indegree_centr_fraction instead')
    centrality_info = file_loader.fetch_json(centrality_file_path)
    centr_by_node = centrality_info[full_centrality_name]
    nodes_degree = 0.0
    for node in attacked_nodes:
        nodes_degree += centr_by_node[node]
    tot_degree = centrality_info['total_' + full_centrality_name]
    fraction = sf.percent_of_part(nodes_degree, tot_degree)
    return fraction


def calc_degree_centr_fraction(G, nodes):
    tot_degree = sum(G.degree(G.nodes()).values())
    nodes_degree = sum(G.degree(nodes).values())
    fraction = sf.percent_of_part(nodes_degree, tot_degree)
    return fraction


def calc_indegree_centr_fraction(G, nodes):
    tot_indegree = sum(G.in_degree(G.nodes()).values())
    nodes_indegree = sum(G.in_degree(nodes).values())
    fraction = sf.percent_of_part(nodes_indegree, tot_indegree)
    return fraction


# this function will be called from another script, each time with a different configuration fpath
def run(conf_fpath, floader):
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
    # A = nx.read_graphml(netw_a_fpath_in, node_type=str)
    A = floader.fetch_graphml(netw_a_fpath_in, str)

    netw_b_fname = config.get('paths', 'netw_b_fname')
    netw_b_fpath_in = os.path.join(netw_dir, netw_b_fname)
    # B = nx.read_graphml(netw_b_fpath_in, node_type=str)
    B = floader.fetch_graphml(netw_b_fpath_in, str)

    netw_inter_fname = config.get('paths', 'netw_inter_fname')
    netw_inter_fpath_in = os.path.join(netw_dir, netw_inter_fname)
    # I = nx.read_graphml(netw_inter_fpath_in, node_type=str)
    I = floader.fetch_graphml(netw_inter_fpath_in, str)

    # if the union graph is needed for this simulation, it's better to have it created in advance and just load it
    if config.has_option('paths', 'netw_union_fname'):
        netw_union_fname = config.get('paths', 'netw_union_fname')
        netw_union_fpath_in = os.path.join(netw_dir, netw_union_fname)
        # ab_union = nx.read_graphml(netw_union_fpath_in, node_type=str)
        ab_union = floader.fetch_graphml(netw_union_fpath_in, str)
    else:
        ab_union = None

    # read run options

    attacked_netw = config.get('run_opts', 'attacked_netw')
    attack_tactic = config.get('run_opts', 'attack_tactic')
    if attack_tactic not in ['targeted', 'ith_node']:
        attack_cnt = config.getint('run_opts', 'attacks')
    intra_support_type = config.get('run_opts', 'intra_support_type')
    if intra_support_type == 'cluster_size':
        min_cluster_size = config.getint('run_opts', 'min_cluster_size')
    inter_support_type = config.get('run_opts', 'inter_support_type')

    if attacked_netw == A.graph['name']:
        attacked_G = A
    elif attacked_netw == B.graph['name']:
        attacked_G = B
    elif attacked_netw.lower() == 'both':
        if ab_union is None:
            raise ValueError('A union graph is needed, specify "netw_union_fname" and make sure the file exists')
        attacked_G = ab_union
    else:
        raise ValueError('Invalid value for parameter "attacked_netw": ' + attacked_netw)

    attacks_for_A_only = ['most_inter_used_distr_subs', 'most_intra_used_distr_subs', 'most_intra_used_transm_subs',
                          'most_intra_used_generators']
    if attack_tactic in attacks_for_A_only and attacked_netw != A.graph['name']:
        raise ValueError('Attack {} can only be used on the power network A'.format(attack_tactic))

    usage_attacks = ['most_inter_used_distr_subs', 'most_intra_used_distr_subs', 'most_intra_used_transm_subs',
                     'most_intra_used_generators', 'most_inter_used_relays', 'most_intra_used_relays']

    centrality_attacks = ['betweenness_centrality_rank', 'closeness_centrality_rank', 'indegree_centrality_rank',
                          'katz_centrality_rank']

    if attack_tactic in usage_attacks:
        secondary_sort = config.get('run_opts', 'secondary_sort')
        if secondary_sort not in ['random_shuffle', 'sort_by_id']:
            raise ValueError('Invalid value for parameter "secondary_sort": ' + secondary_sort)

    if attack_tactic in centrality_attacks:
        min_rank = config.getint('run_opts', 'min_rank')
        calc_centr_on = config.get('run_opts', 'calc_centrality_on')
        if calc_centr_on.lower() == 'attacked_netw':
            G_for_centr = attacked_G
        elif calc_centr_on.lower() == 'netw_inter':
            # note that calculating centrality on the inter graph is not the same as calculating it on the union graph
            G_for_centr = I
        else:
            raise ValueError('Invalid value for parameter "calc_centr_on": ' + calc_centr_on)

        # load file with precalculated centrality metrics
        centr_fname = 'node_centrality_{}.json'.format(G_for_centr.graph['name'])
        centr_fpath = os.path.join(netw_dir, centr_fname)
        centrality_info = floader.fetch_json(centr_fpath)

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

    if config.has_option('run_opts', 'save_attacked_roles'):
        save_attacked_roles = config.getboolean('run_opts', 'save_attacked_roles')
    else:
        save_attacked_roles = False

    # ensure output directories exist and are empty
    sf.ensure_dir_exists(results_dir)
    sf.ensure_dir_exists(os.path.dirname(end_stats_fpath))

    # read information used to identify this simulation run
    sim_group = config.getint('misc', 'sim_group')
    instance = config.getint('misc', 'instance')

    if config.has_option('paths', 'batch_conf_fpath'):
        batch_conf_fpath = config.get('paths', 'batch_conf_fpath')
    else:
        batch_conf_fpath = ''

    if config.has_option('misc', 'run'):
        run_num = config.getint('misc', 'run')
    else:
        run_num = 0

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

        if attack_tactic == 'random':
            attacked_nodes = choose_random_nodes(attacked_G, attack_cnt, seed)
        elif attack_tactic == 'ith_node':
            node_rank = config.getint('run_opts', 'node_rank')
            attacked_nodes = [pick_ith_node(attacked_G, node_rank)]
        elif attack_tactic == 'targeted':
            target_nodes = config.get('run_opts', 'target_nodes')
            # TODO: understand why this returns unicode even if it should not
            if sys.version_info >= (2, 7, 9):
                target_nodes = str(target_nodes)
            attacked_nodes = [node for node in target_nodes.split()]  # split list on space
        # TODO: use choose_nodes_by_rank, throw away _centrality_rank columns
        elif attack_tactic == 'betweenness_centrality_rank':
            ranked_nodes = centrality_info['betweenness_centrality_rank']
            attacked_nodes = pick_nodes_by_rank(ranked_nodes, attack_cnt, min_rank)
        elif attack_tactic == 'closeness_centrality_rank':
            ranked_nodes = centrality_info['closeness_centrality_rank']
            attacked_nodes = pick_nodes_by_rank(ranked_nodes, attack_cnt, min_rank)
        elif attack_tactic == 'indegree_centrality_rank':
            ranked_nodes = centrality_info['indegree_centrality_rank']
            attacked_nodes = pick_nodes_by_rank(ranked_nodes, attack_cnt, min_rank)
        elif attack_tactic == 'katz_centrality_rank':
            ranked_nodes = centrality_info['katz_centrality_rank']
            attacked_nodes = pick_nodes_by_rank(ranked_nodes, attack_cnt, min_rank)
        elif attack_tactic == 'most_inter_used_distr_subs':
            attacked_nodes = choose_most_inter_used_nodes(attacked_G, I, attack_cnt, 'distribution_substation',
                                                          secondary_sort, seed)
        elif attack_tactic == 'most_intra_used_distr_subs':
            attacked_nodes = choose_most_intra_used_nodes(attacked_G, attack_cnt, 'distribution_substation',
                                                          secondary_sort, seed)
        elif attack_tactic == 'most_intra_used_transm_subs':
            attacked_nodes = choose_most_intra_used_nodes(attacked_G, attack_cnt, 'transmission_substation',
                                                          secondary_sort, seed)
        elif attack_tactic == 'most_intra_used_generators':
            attacked_nodes = choose_most_intra_used_nodes(attacked_G, attack_cnt, 'generator', secondary_sort, seed)
        else:
            raise ValueError('Invalid value for parameter "attack_tactic": ' + attack_tactic)

        attacked_nodes_a = []
        attacked_nodes_b = []
        for node in attacked_nodes:
            node_netw = I.node[node]['network']
            if node_netw == A.graph['name']:
                attacked_nodes_a.append(node)
            elif node_netw == B.graph['name']:
                attacked_nodes_b.append(node)
            else:
                raise RuntimeError('Node {} network "{}" marked in inter graph is neither A nor B'.format(
                    node, node_netw))

        node_cnt_a = A.number_of_nodes()
        p_atkd_a = sf.percent_of_part(len(attacked_nodes_a), node_cnt_a)

        node_cnt_b = B.number_of_nodes()
        p_atkd_b = sf.percent_of_part(len(attacked_nodes_b), node_cnt_b)

        atkd_cnt = len(attacked_nodes_a) + len(attacked_nodes_b)
        p_atkd = sf.percent_of_part(atkd_cnt, node_cnt_a + node_cnt_b)

        if ml_stats_fpath:  # if this string is not empty

            # fractions of connectivity lost due to the attacks
            lost_deg_a = calc_degree_centr_fraction(A, attacked_nodes_a)
            lost_deg_b = calc_degree_centr_fraction(B, attacked_nodes_b)
            lost_indeg_i = calc_indegree_centr_fraction(I, attacked_nodes_a + attacked_nodes_b)
            centr_fpath = os.path.join(netw_dir, 'node_centrality_general.json')
            lost_relay_betw = calc_centrality_fraction(attacked_nodes, 'relay_betweenness_centrality', floader,
                                                       centr_fpath)
            lost_ts_betw = calc_centrality_fraction(attacked_nodes, 'transm_subst_betweenness_centrality', floader,
                                                    centr_fpath)

            # Please note that this section only works for nodes created using realistic roles
            if save_attacked_roles is True:
                # count how many nodes there are for each role in the power network
                nodes_a_by_role = {'generator': 0, 'transmission_substation': 0, 'distribution_substation': 0}
                try:
                    for node in A.nodes():
                        node_role = A.node[node]['role']
                        nodes_a_by_role[node_role] += 1
                except KeyError:
                    raise RuntimeError('Unrecognized node role {} in graph {}'.format(node_role, A.graph['name']))

                # count how many nodes were attacked for each power role
                atkd_nodes_a_by_role = {'generator': 0, 'transmission_substation': 0, 'distribution_substation': 0}
                for node in attacked_nodes_a:
                    node_role = A.node[node]['role']
                    atkd_nodes_a_by_role[node_role] += 1

                # calculate the percentage of nodes attacked for each power role
                p_atkd_gen = sf.percent_of_part(atkd_nodes_a_by_role['generator'], nodes_a_by_role['generator'])
                p_atkd_ts = sf.percent_of_part(atkd_nodes_a_by_role['transmission_substation'],
                                               nodes_a_by_role['transmission_substation'])
                p_atkd_ds = sf.percent_of_part(atkd_nodes_a_by_role['distribution_substation'],
                                               nodes_a_by_role['distribution_substation'])

                # count how many nodes there are for each role in the telecom network
                nodes_b_by_role = {'relay': 0, 'controller': 0}
                try:
                    for node in B.nodes():
                        node_role = B.node[node]['role']
                        nodes_b_by_role[node_role] += 1
                except KeyError:
                    raise RuntimeError('Unrecognized node role {} in graph {}'.format(node_role, B.graph['name']))

                # count how many nodes were attacked for each telecom role
                atkd_nodes_b_by_role = {'relay': 0, 'controller': 0}
                for node in attacked_nodes_b:
                    node_role = B.node[node]['role']
                    atkd_nodes_b_by_role[node_role] += 1

                # calculate the percentage of nodes attacked for each telecom role
                p_atkd_rel = sf.percent_of_part(atkd_nodes_b_by_role['relay'], nodes_b_by_role['relay'])
                p_atkd_cc = sf.percent_of_part(atkd_nodes_b_by_role['controller'], nodes_b_by_role['controller'])

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
        end_stats_header = ['batch_conf_fpath', 'sim_group', 'instance', '#run', '#dead', '#dead_a', '#dead_b']
        if save_attacked_roles is True:
            end_stats_header.extend(['#atkd_gen', '#atkd_ts', '#atkd_ds', '#atkd_rel', '#atkd_cc'])
        if save_death_cause is True:
            end_stats_header.extend(['no_intra_sup_a', 'no_inter_sup_a',
                                     'no_intra_sup_b', 'no_inter_sup_b'])
            if inter_support_type == 'realistic':
                end_stats_header.extend(['no_sup_ccs', 'no_sup_relays', 'no_com_path'])

        end_stats = csv.DictWriter(end_stats_file, end_stats_header, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        if write_header is True:
            end_stats.writeheader()

        end_stats_row = {'batch_conf_fpath': batch_conf_fpath, 'sim_group': sim_group, 'instance': instance,
                         '#run': run_num, '#dead': total_dead_a + total_dead_b,
                         '#dead_a': total_dead_a, '#dead_b': total_dead_b}
        if save_attacked_roles is True:
            end_stats_row.update({'#atkd_gen': atkd_nodes_a_by_role['generator'],
                                  '#atkd_ts': atkd_nodes_a_by_role['transmission_substation'],
                                  '#atkd_ds': atkd_nodes_a_by_role['distribution_substation'],
                                  '#atkd_rel': atkd_nodes_b_by_role['relay'],
                                  '#atkd_cc': atkd_nodes_b_by_role['controller']})
        if save_death_cause is True:
            end_stats_row.update({'no_intra_sup_a': intra_sup_deaths_a, 'no_inter_sup_a': inter_sup_deaths_a,
                                  'no_intra_sup_b': intra_sup_deaths_b, 'no_inter_sup_b': inter_sup_deaths_b})
            if inter_support_type == 'realistic':
                end_stats_row.update({'no_sup_ccs': no_sup_ccs_deaths, 'no_sup_relays': no_sup_relays_deaths,
                                      'no_com_path': no_com_path_deaths})
        end_stats.writerow(end_stats_row)

    if ml_stats_fpath:  # if this string is not empty

        # percentages of dead nodes over the initial number of nodes in the graph
        p_dead_a = sf.percent_of_part(total_dead_a, node_cnt_a)
        p_dead_b = sf.percent_of_part(total_dead_b, node_cnt_b)
        p_dead = sf.percent_of_part(total_dead_a + total_dead_b, node_cnt_a + node_cnt_b)

        # calculate the overall centrality of the attacked nodes

        all_centr_stats = {}

        centr_name = 'betweenness_centrality'
        centr_fpath_a = os.path.join(netw_dir, 'node_centrality_{}.json'.format(A.graph['name']))
        centr_stats = calc_stats_on_centrality(attacked_nodes_a, centr_name, 'betw_c_a', floader, centr_fpath_a)
        all_centr_stats.update(centr_stats)
        centr_fpath_b = os.path.join(netw_dir, 'node_centrality_{}.json'.format(B.graph['name']))
        centr_stats = calc_stats_on_centrality(attacked_nodes_b, centr_name, 'betw_c_b', floader, centr_fpath_b)
        all_centr_stats.update(centr_stats)
        tot_node_cnt = node_cnt_a + node_cnt_b
        centr_fpath_ab = os.path.join(netw_dir, 'node_centrality_{}.json'.format(ab_union.graph['name']))
        centr_stats = calc_stats_on_centrality(attacked_nodes, centr_name, 'betw_c_ab', floader, centr_fpath_ab)
        all_centr_stats.update(centr_stats)
        centr_fpath_i = os.path.join(netw_dir, 'node_centrality_{}.json'.format(I.graph['name']))
        centr_stats = calc_stats_on_centrality(attacked_nodes, centr_name, 'betw_c_i', floader, centr_fpath_i)
        all_centr_stats.update(centr_stats)

        centr_name = 'closeness_centrality'
        centr_stats = calc_stats_on_centrality(attacked_nodes_a, centr_name, 'clos_c_a', floader, centr_fpath_a)
        all_centr_stats.update(centr_stats)
        centr_stats = calc_stats_on_centrality(attacked_nodes_b, centr_name, 'clos_c_b', floader, centr_fpath_b)
        all_centr_stats.update(centr_stats)
        centr_stats = calc_stats_on_centrality(attacked_nodes, centr_name, 'clos_c_ab', floader, centr_fpath_ab)
        all_centr_stats.update(centr_stats)
        centr_stats = calc_stats_on_centrality(attacked_nodes, centr_name, 'clos_c_i', floader, centr_fpath_i)
        all_centr_stats.update(centr_stats)

        centr_name = 'indegree_centrality'
        centr_stats = calc_stats_on_centrality(attacked_nodes, centr_name, 'indeg_c_ab', floader, centr_fpath_ab)
        all_centr_stats.update(centr_stats)
        centr_stats = calc_stats_on_centrality(attacked_nodes, centr_name, 'indeg_c_i', floader, centr_fpath_i)
        all_centr_stats.update(centr_stats)

        centr_name = 'katz_centrality'
        centr_stats = calc_stats_on_centrality(attacked_nodes, centr_name, 'katz_c_ab', floader, centr_fpath_ab)
        all_centr_stats.update(centr_stats)
        centr_stats = calc_stats_on_centrality(attacked_nodes, centr_name, 'katz_c_i', floader, centr_fpath_i)
        all_centr_stats.update(centr_stats)

        centr_name = 'relay_betweenness_centrality'
        centr_fpath_gen = os.path.join(netw_dir, 'node_centrality_general.json')
        centr_stats = calc_stats_on_centrality(attacked_nodes, centr_name, 'rel_betw_c', floader, centr_fpath_gen)
        all_centr_stats.update(centr_stats)

        centr_name = 'transm_subst_betweenness_centrality'
        centr_stats = calc_stats_on_centrality(attacked_nodes, centr_name, 'ts_betw_c', floader, centr_fpath_gen)
        all_centr_stats.update(centr_stats)

        # if we need to create a new file, then remember to add the header
        if os.path.isfile(ml_stats_fpath) is False:
            write_header = True
        else:
            write_header = False

        with open(ml_stats_fpath, 'ab') as ml_stats_file:
            ml_stats_header = ['batch_conf_fpath', 'sim_group', 'instance', '#run', 'seed', '#atkd',
                               'p_atkd', 'p_dead', 'p_atkd_a', 'p_dead_a', 'p_atkd_b', 'p_dead_b']

            # add statistics about centrality, sorted so they can be more found easily in the output file
            ml_stats_header.extend(sorted(all_centr_stats.keys(), key=sf.natural_sort_key))

            if save_attacked_roles is True:
                ml_stats_header.extend(['p_atkd_gen', 'p_atkd_ts', 'p_atkd_ds', 'p_atkd_rel', 'p_atkd_cc'])

            ml_stats_header.extend(['lost_deg_a', 'lost_deg_b', 'lost_indeg_i', 'lost_rel_betw', 'lost_ts_betw'])

            if save_death_cause is True:
                ml_stats_header.extend(['p_no_intra_sup_a', 'p_no_inter_sup_a',
                                        'p_no_intra_sup_b', 'p_no_inter_sup_b'])
                if inter_support_type == 'realistic':
                    ml_stats_header.extend(['p_no_sup_ccs', 'p_no_sup_relays', 'p_no_com_path'])

            ml_stats = csv.DictWriter(ml_stats_file, ml_stats_header, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            if write_header is True:
                ml_stats.writeheader()

            ml_stats_row = {'batch_conf_fpath': batch_conf_fpath, 'sim_group': sim_group, 'instance': instance,
                            '#run': run_num, 'seed': seed, '#atkd': atkd_cnt,
                            'p_atkd': p_atkd, 'p_atkd_a': p_atkd_a, 'p_atkd_b': p_atkd_b,
                            'p_dead': p_dead, 'p_dead_a': p_dead_a, 'p_dead_b': p_dead_b}

            ml_stats_row.update(all_centr_stats)

            if save_attacked_roles is True:
                ml_stats_row.update({'p_atkd_gen': p_atkd_gen, 'p_atkd_ts': p_atkd_ts, 'p_atkd_ds': p_atkd_ds,
                                     'p_atkd_rel': p_atkd_rel, 'p_atkd_cc': p_atkd_cc,})

            ml_stats_row.update({'lost_deg_a': lost_deg_a, 'lost_deg_b': lost_deg_b, 'lost_indeg_i': lost_indeg_i,
                                 'lost_rel_betw': lost_relay_betw, 'lost_ts_betw': lost_ts_betw})

            if save_death_cause is True:
                p_no_intra_sup_a = sf.percent_of_part(intra_sup_deaths_a, total_dead_a)
                p_no_inter_sup_a = sf.percent_of_part(inter_sup_deaths_a, total_dead_a)
                p_no_intra_sup_b = sf.percent_of_part(intra_sup_deaths_b, total_dead_b)
                p_no_inter_sup_b = sf.percent_of_part(inter_sup_deaths_b, total_dead_b)
                ml_stats_row.update({'p_no_intra_sup_a': p_no_intra_sup_a, 'p_no_inter_sup_a': p_no_inter_sup_a,
                                     'p_no_intra_sup_b': p_no_intra_sup_b, 'p_no_inter_sup_b': p_no_inter_sup_b})

                if inter_support_type == 'realistic':
                    p_no_sup_ccs = sf.percent_of_part(no_sup_ccs_deaths, total_dead_a)
                    p_no_sup_relays = sf.percent_of_part(no_sup_relays_deaths, total_dead_a)
                    p_no_com_path = sf.percent_of_part(no_com_path_deaths, total_dead_a)
                    ml_stats_row.update({'p_no_sup_ccs': p_no_sup_ccs, 'p_no_sup_relays': p_no_sup_relays,
                                         'p_no_com_path': p_no_com_path})

            ml_stats.writerow(ml_stats_row)
