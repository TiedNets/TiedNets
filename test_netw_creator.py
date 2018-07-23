import os
import shutil
import networkx as nx
from collections import defaultdict
import netw_creator as nc
import shared_functions as sf

__author__ = 'Agostino Sturaro'

this_dir = os.path.normpath(os.path.dirname(__file__))
logging_conf_fpath = os.path.join(this_dir, 'logging_base_conf.json')


def test_assign_power_roles():
    G = nx.Graph()
    G.add_nodes_from(list(range(0, 15)))
    H = G.copy()

    nc.assign_power_roles(G, 4, 6, 5, False, 128)
    nodes_by_role_G = defaultdict(set)
    for node in G.nodes():
        role = G.node[node]['role']
        nodes_by_role_G[role].add(node)

    nc.assign_power_roles(H, 5, 3, 7, False, 128)
    nodes_by_role_H = defaultdict(set)
    for node in H.nodes():
        role = H.node[node]['role']
        nodes_by_role_H[role].add(node)

    assert len(nodes_by_role_G['generator']) == 4
    assert len(nodes_by_role_G['distribution_substation']) == 5
    assert len(nodes_by_role_H['generator']) == 5
    assert len(nodes_by_role_H['distribution_substation']) == 7
    assert set.issuperset(nodes_by_role_H['generator'], nodes_by_role_G['generator'])
    assert set.issuperset(nodes_by_role_H['distribution_substation'], nodes_by_role_G['distribution_substation'])


def test_assign_power_roles_to_subnets():
    G = nx.Graph()
    G.add_nodes_from(list(range(0, 5)), subnet=0)
    G.add_nodes_from(list(range(5, 10)), subnet=1)
    G.add_nodes_from(list(range(10, 15)), subnet=2)
    H = G.copy()

    nc.assign_power_roles_to_subnets(G, 4, 5, 128)
    nodes_by_role_G = defaultdict(set)
    for node in G.nodes():
        role = G.node[node]['role']
        nodes_by_role_G[role].add(node)
    assert (nodes_by_role_G)

    nc.assign_power_roles_to_subnets(H, 5, 7, 128)
    nodes_by_role_H = defaultdict(set)
    for node in H.nodes():
        role = H.node[node]['role']
        nodes_by_role_H[role].add(node)
    assert (nodes_by_role_H)

    assert len(nodes_by_role_G['generator']) == 4
    assert len(nodes_by_role_G['distribution_substation']) == 5
    assert len(nodes_by_role_H['generator']) == 5
    assert len(nodes_by_role_H['distribution_substation']) == 7
    assert set.issuperset(nodes_by_role_H['generator'], nodes_by_role_G['generator'])
    assert set.issuperset(nodes_by_role_H['distribution_substation'], nodes_by_role_G['distribution_substation'])


def test_create_k_to_n_dep():
    nodes_a = [
        ('a0', {'x': 0, 'y': 0, 'role': 'power'}),
        ('a1', {'x': 0, 'y': 1, 'role': 'power'}),
        ('a2', {'x': 0, 'y': 2, 'role': 'power'}),
        ('a3', {'x': 0, 'y': 3, 'role': 'power'})
    ]

    nodes_a_with_roles = [
        ('D0', {'x': 0, 'y': 0, 'role': 'distribution_substation'}),
        ('G0', {'x': 0, 'y': 1, 'role': 'generator'}),
        ('T0', {'x': 0, 'y': 2, 'role': 'transmission_substation'}),
        ('D1', {'x': 0, 'y': 3, 'role': 'distribution_substation'})
    ]

    nodes_b_1cc = [
        ('C0', {'x': 1, 'y': 0, 'role': 'controller'}),
        ('R0', {'x': 1, 'y': 1, 'role': 'relay'}),
        ('R1', {'x': 1, 'y': 2, 'role': 'relay'}),
        ('C1', {'x': 1, 'y': 3, 'role': 'controller'})
    ]

    nodes_b_2cc = [
        ('C0', {'x': 1, 'y': 0, 'role': 'controller'}),
        ('R0', {'x': 1, 'y': 1, 'role': 'relay'}),
        ('R1', {'x': 1, 'y': 2, 'role': 'relay'}),
        ('R2', {'x': 1, 'y': 3, 'role': 'relay'})
    ]

    A = nx.DiGraph()
    A.add_nodes_from(nodes_a)

    A_with_roles = nx.DiGraph()
    A_with_roles.add_nodes_from(nodes_a_with_roles)

    B_1cc = nx.DiGraph()
    B_1cc.add_nodes_from(nodes_b_1cc)

    B_2cc = nx.DiGraph()
    B_2cc.add_nodes_from(nodes_b_2cc)

    # k is the number of control centers supporting each power node
    # n is the number of power nodes that each control center supports

    # 1 control center supporting everything

    I = nc.create_k_to_n_dep(A, B_2cc, 1, 4, power_roles=False, prefer_nearest=True)
    wanted_edges = [('a0', 'C0'), ('a0', 'R0'),
                    ('a1', 'C0'), ('a1', 'R0'),
                    ('a2', 'C0'), ('a2', 'R1'),
                    ('a3', 'C0'), ('a3', 'R2'),
                    ('C0', 'a0'), ('R0', 'a1'), ('R1', 'a2'), ('R2', 'a3')]
    assert sorted(I.edges()) == sorted(wanted_edges)

    I = nc.create_k_to_n_dep(A_with_roles, B_2cc, 1, 4, power_roles=True, prefer_nearest=True)
    wanted_edges = [('D0', 'C0'), ('D0', 'R0'),
                    ('G0', 'C0'), ('G0', 'R0'),
                    ('T0', 'C0'), ('T0', 'R1'),
                    ('D1', 'C0'), ('D1', 'R2'),
                    ('C0', 'D0'), ('R0', 'D0'), ('R1', 'D1'), ('R2', 'D1')]
    assert sorted(I.edges()) == sorted(wanted_edges)

    I_0 = nc.create_k_to_n_dep(A, B_2cc, 1, 4, power_roles=False, prefer_nearest=False, seed=128)
    I_1 = nc.create_k_to_n_dep(A, B_2cc, 1, 4, power_roles=False, prefer_nearest=False, seed=128)
    wanted_out_degrees = {'a0': 2, 'a1': 2, 'a2': 2, 'a3': 2, 'C0': 1, 'R0': 1, 'R1': 1, 'R2': 1}
    assert I_0.out_degree(['a0', 'a1', 'a2', 'a3', 'C0', 'R0', 'R1', 'R2']) == wanted_out_degrees
    assert sum(I_0.in_degree(['a0', 'a1', 'a2', 'a3']).values()) == 4
    assert sum(I_0.in_degree(['R0', 'R1', 'R2']).values()) == 4
    assert sum(I_0.in_degree(['C0']).values()) == 4
    assert sorted(I_0.edges()) == sorted(I_1.edges())

    I_0 = nc.create_k_to_n_dep(A_with_roles, B_2cc, 1, 4, power_roles=True, prefer_nearest=False, seed=128)
    I_1 = nc.create_k_to_n_dep(A_with_roles, B_2cc, 1, 4, power_roles=True, prefer_nearest=False, seed=128)
    wanted_out_degrees = {'D0': 2, 'G0': 2, 'T0': 2, 'D1': 2, 'C0': 1, 'R0': 1, 'R1': 1, 'R2': 1}
    assert I_0.out_degree(['D0', 'G0', 'T0', 'D1', 'C0', 'R0', 'R1', 'R2']) == wanted_out_degrees
    assert sum(I_0.in_degree(['D0', 'G0', 'T0', 'D1']).values()) == 4
    assert sum(I_0.in_degree(['R0', 'R1', 'R2']).values()) == 4
    assert sum(I_0.in_degree(['C0']).values()) == 4
    assert sorted(I_0.edges()) == sorted(I_1.edges())

    # 2 control centers each supporting all power nodes

    I = nc.create_k_to_n_dep(A, B_1cc, 1, 2, power_roles=False, prefer_nearest=True)
    wanted_edges = [('a0', 'C0'), ('a0', 'R0'),
                    ('a1', 'C0'), ('a1', 'R0'),
                    ('a2', 'C1'), ('a2', 'R1'),
                    ('a3', 'C1'), ('a3', 'R1'),
                    ('C0', 'a0'), ('R0', 'a1'), ('R1', 'a2'), ('C1', 'a3')]
    assert sorted(I.edges()) == sorted(wanted_edges)

    I = nc.create_k_to_n_dep(A_with_roles, B_1cc, 1, 2, power_roles=True, prefer_nearest=True)
    wanted_edges = [('D0', 'C0'), ('D0', 'R0'),
                    ('G0', 'C0'), ('G0', 'R0'),
                    ('T0', 'C1'), ('T0', 'R1'),
                    ('D1', 'C1'), ('D1', 'R1'),
                    ('C0', 'D0'), ('R0', 'D0'), ('R1', 'D1'), ('C1', 'D1')]
    assert sorted(I.edges()) == sorted(wanted_edges)

    I_0 = nc.create_k_to_n_dep(A, B_1cc, 1, 2, power_roles=False, prefer_nearest=False, seed=128)
    I_1 = nc.create_k_to_n_dep(A, B_1cc, 1, 2, power_roles=False, prefer_nearest=False, seed=128)
    wanted_out_degrees = {'a0': 2, 'a1': 2, 'a2': 2, 'a3': 2, 'C0': 1, 'R0': 1, 'R1': 1, 'C1': 1}
    assert I_0.out_degree(['a0', 'a1', 'a2', 'a3', 'C0', 'R0', 'R1', 'C1']) == wanted_out_degrees
    assert sum(I_0.in_degree(['a0', 'a1', 'a2', 'a3']).values()) == 4
    assert sum(I_0.in_degree(['R0', 'R1']).values()) == 4
    assert sum(I_0.in_degree(['C0', 'C1']).values()) == 4
    assert sorted(I_0.edges()) == sorted(I_1.edges())

    I_0 = nc.create_k_to_n_dep(A_with_roles, B_1cc, 1, 2, power_roles=True, prefer_nearest=False, seed=128)
    I_1 = nc.create_k_to_n_dep(A_with_roles, B_1cc, 1, 2, power_roles=True, prefer_nearest=False, seed=128)
    wanted_out_degrees = {'D0': 2, 'G0': 2, 'T0': 2, 'D1': 2, 'C0': 1, 'R0': 1, 'R1': 1, 'C1': 1}
    assert I_0.out_degree(['D0', 'G0', 'T0', 'D1', 'C0', 'R0', 'R1', 'C1']) == wanted_out_degrees
    assert sum(I_0.in_degree(['D0', 'G0', 'T0', 'D1']).values()) == 4
    assert sum(I_0.in_degree(['R0', 'R1']).values()) == 4
    assert sum(I_0.in_degree(['C0', 'C1']).values()) == 4
    assert sorted(I_0.edges()) == sorted(I_1.edges())

    # 2 control centers each supporting half of the power network

    I = nc.create_k_to_n_dep(A, B_1cc, 2, 4, power_roles=False, prefer_nearest=True)
    wanted_edges = [('a0', 'C0'), ('a0', 'C1'), ('a0', 'R0'),
                    ('a1', 'C0'), ('a1', 'C1'), ('a1', 'R0'),
                    ('a2', 'C0'), ('a2', 'C1'), ('a2', 'R1'),
                    ('a3', 'C0'), ('a3', 'C1'), ('a3', 'R1'),
                    ('C0', 'a0'), ('R0', 'a1'), ('R1', 'a2'), ('C1', 'a3')]
    assert sorted(I.edges()) == sorted(wanted_edges)

    I = nc.create_k_to_n_dep(A_with_roles, B_1cc, 2, 4, power_roles=True, prefer_nearest=True)
    wanted_edges = [('D0', 'C0'), ('D0', 'C1'), ('D0', 'R0'),
                    ('G0', 'C0'), ('G0', 'C1'), ('G0', 'R0'),
                    ('T0', 'C0'), ('T0', 'C1'), ('T0', 'R1'),
                    ('D1', 'C0'), ('D1', 'C1'), ('D1', 'R1'),
                    ('C0', 'D0'), ('R0', 'D0'), ('R1', 'D1'), ('C1', 'D1')]
    assert sorted(I.edges()) == sorted(wanted_edges)

    I_0 = nc.create_k_to_n_dep(A, B_1cc, 2, 4, power_roles=False, prefer_nearest=False, seed=128)
    I_1 = nc.create_k_to_n_dep(A, B_1cc, 2, 4, power_roles=False, prefer_nearest=False, seed=128)
    wanted_out_degrees = {'a0': 3, 'a1': 3, 'a2': 3, 'a3': 3, 'C0': 1, 'R0': 1, 'R1': 1, 'C1': 1}
    assert I_0.out_degree(['a0', 'a1', 'a2', 'a3', 'C0', 'R0', 'R1', 'C1']) == wanted_out_degrees
    assert sum(I_0.in_degree(['a0', 'a1', 'a2', 'a3']).values()) == 4
    assert sum(I_0.in_degree(['R0', 'R1']).values()) == 4
    assert sum(I_0.in_degree(['C0', 'C1']).values()) == 8
    assert sorted(I_0.edges()) == sorted(I_1.edges())

    I_0 = nc.create_k_to_n_dep(A_with_roles, B_1cc, 2, 4, power_roles=True, prefer_nearest=False, seed=128)
    I_1 = nc.create_k_to_n_dep(A_with_roles, B_1cc, 2, 4, power_roles=True, prefer_nearest=False, seed=128)
    wanted_out_degrees = {'D0': 3, 'G0': 3, 'T0': 3, 'D1': 3, 'C0': 1, 'R0': 1, 'R1': 1, 'C1': 1}
    assert I_0.out_degree(['D0', 'G0', 'T0', 'D1', 'C0', 'R0', 'R1', 'C1']) == wanted_out_degrees
    assert sum(I_0.in_degree(['D0', 'G0', 'T0', 'D1']).values()) == 4
    assert sum(I_0.in_degree(['R0', 'R1']).values()) == 4
    assert sum(I_0.in_degree(['C0', 'C1']).values()) == 8
    assert sorted(I_0.edges()) == sorted(I_1.edges())


def test_count_node_disjoint_paths():
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (3, 5), (3, 6)]
    G = nx.Graph(edges)
    path_cnt = nc.count_node_disjoint_paths(G, 1, 2)
    assert path_cnt == 2
    path_cnt = nc.count_node_disjoint_paths(G, 4, 6)
    assert path_cnt == 2


def test_calc_relay_betweenness():
    A = nx.Graph()
    nodes_a = [
        ('D0', {'role': 'distribution_substation'}),
        ('T0', {'role': 'transmission_substation'}),
        ('G0', {'role': 'generator'})
    ]
    A.add_nodes_from(nodes_a)
    edges_a = [('G0', 'T0'), ('T0', 'D0')]
    A.add_edges_from(edges_a)

    B = nx.Graph()
    nodes_b = [
        ('C0', {'role': 'controller'}),
        ('R0', {'role': 'relay'}),
        ('R1', {'role': 'relay'})
    ]
    B.add_nodes_from(nodes_b)
    edges_b = [('R0', 'R1'), ('R1', 'C0')]
    B.add_edges_from(edges_b)

    I = nx.DiGraph()
    I.add_nodes_from(nodes_a + nodes_b)
    edges_i = [
        ('D0', 'C0'), ('T0', 'C0'), ('G0', 'C0'),
        ('D0', 'R0'), ('T0', 'R0'), ('G0', 'R0')
    ]
    I.add_edges_from(edges_i)

    results = nc.calc_relay_betweenness(A, B, I)
    expected = {'R0': 1.0, 'R1': 1.0, 'C0': 0.0, 'D0': 0.0, 'T0': 0.0, 'G0': 0.0}
    assert results == expected

    B.remove_edges_from(B.edges())
    edges_b = [('R0', 'C0'), ('R1', 'C0')]
    B.add_edges_from(edges_b)

    I.remove_edges_from(I.edges())
    edges_i = [
        ('D0', 'C0'), ('T0', 'C0'), ('G0', 'C0'),
        ('D0', 'R0'), ('T0', 'R0'), ('G0', 'R1')
    ]
    I.add_edges_from(edges_i)

    results = nc.calc_relay_betweenness(A, B, I)
    expected = {'R0': 2./3, 'R1': 1./3, 'C0': 0.0, 'D0': 0.0, 'T0': 0.0, 'G0': 0.0}
    assert results == expected

    del B
    B = nx.Graph()
    nodes_b = [
        ('C0', {'role': 'controller'}),
        ('C1', {'role': 'controller'}),
        ('R0', {'role': 'relay'}),
        ('R1', {'role': 'relay'}),
        ('R2', {'role': 'relay'}),
        ('R3', {'role': 'relay'})
    ]
    B.add_nodes_from(nodes_b)
    edges_b = [('R0', 'C0'), ('R0', 'R1'), ('R1', 'R2'), ('R2', 'C1'), ('R2', 'R3')]
    B.add_edges_from(edges_b)

    del I
    I = nx.DiGraph()
    I.add_nodes_from(nodes_a + nodes_b)
    edges_i = [
        ('D0', 'C0'), ('T0', 'C1'), ('G0', 'C1'),
        ('D0', 'R0'), ('T0', 'R1'), ('G0', 'R2')
    ]
    I.add_edges_from(edges_i)

    results = nc.calc_relay_betweenness(A, B, I)
    expected = {'R0': 1. / 3, 'R1': 1. / 3, 'R2': 2. / 3, 'R3': 0., 'C0': 0.0, 'C1': 0.0,
                'D0': 0.0, 'T0': 0.0, 'G0': 0.0}
    assert results == expected

    I.add_edge('T0', 'C0')
    results = nc.calc_relay_betweenness(A, B, I)
    expected = {'R0': 0.5, 'R1': 0.5, 'R2': 0.5, 'R3': 0., 'C0': 0.0, 'C1': 0.0,
                'D0': 0.0, 'T0': 0.0, 'G0': 0.0}
    assert results == expected

    B.remove_edges_from(B.edges())
    edges_b = [('R0', 'C0'), ('R0', 'R1'), ('R2', 'R3'), ('R3', 'C1')]
    B.add_edges_from(edges_b)

    I.remove_edges_from(I.edges())
    edges_i = [
        ('D0', 'C0'), ('T0', 'C0'), ('T0', 'C1'), ('G0', 'C1'),
        ('D0', 'R0'), ('T0', 'R0'), ('T0', 'R1'), ('T0', 'R2'), ('T0', 'R3'), ('G0', 'R3')
    ]
    I.add_edges_from(edges_i)

    # here we see that this is is a peculiar betweenness
    results = nc.calc_relay_betweenness(A, B, I)
    expected = {'R0': 3. / 6, 'R1': 1. / 6, 'R2': 1. / 6, 'R3': 3. / 6, 'C0': 0.0, 'C1': 0.0,
                'D0': 0.0, 'T0': 0.0, 'G0': 0.0}
    assert results == expected

    B.add_node('R4', {'role': 'relay'})  # an isolated node
    I.add_edge('G0', 'R4')  # a useless dependency
    results = nc.calc_relay_betweenness(A, B, I)
    expected = {'R0': 3. / 6, 'R1': 1. / 6, 'R2': 1. / 6, 'R3': 3. / 6, 'R4': 0., 'C0': 0.0, 'C1': 0.0,
                'D0': 0.0, 'T0': 0.0, 'G0': 0.0}
    assert results == expected


def test_calc_transm_subst_betweenness():
    A = nx.Graph()
    nodes_a = [
        ('D0', {'role': 'distribution_substation'}),
        ('T0', {'role': 'transmission_substation'}),
        ('G0', {'role': 'generator'})
    ]
    A.add_nodes_from(nodes_a)
    edges_a = [('G0', 'T0'), ('T0', 'D0')]
    A.add_edges_from(edges_a)

    B = nx.Graph()
    nodes_b = [
        ('C0', {'role': 'controller'}),
        ('R0', {'role': 'relay'})
    ]
    B.add_nodes_from(nodes_b)
    edges_b = [('R0', 'C0')]
    B.add_edges_from(edges_b)

    I = nx.DiGraph()
    I.add_nodes_from(nodes_a + nodes_b)
    edges_i = [
        ('D0', 'C0'), ('T0', 'C0'), ('G0', 'C0'),
        ('D0', 'R0'), ('T0', 'R0'), ('G0', 'R0'),
        ('R0', 'D0'), ('C0', 'D0')
    ]
    I.add_edges_from(edges_i)

    results = nc.calc_transm_subst_betweenness(A, B, I)
    expected = {'D0': 0.0, 'T0': 1.0, 'G0': 0.0, 'R0': 0.0, 'C0': 0.0}
    assert results == expected

    del A
    A = nx.Graph()
    nodes_a = [
        ('D0', {'role': 'distribution_substation'}),
        ('D1', {'role': 'distribution_substation'}),
        ('T0', {'role': 'transmission_substation'}),
        ('G0', {'role': 'generator'}),
        ('G1', {'role': 'generator'})
    ]
    A.add_nodes_from(nodes_a)
    edges_a = [('D0', 'T0'), ('D1', 'T0'), ('T0', 'G0'), ('T0', 'G1')]
    A.add_edges_from(edges_a)

    I = nx.DiGraph()
    I.add_nodes_from(nodes_a + nodes_b)
    edges_i = [
        ('D0', 'C0'), ('T0', 'C0'), ('G0', 'C0'), ('D1', 'C0'), ('G1', 'C0'),
        ('D0', 'R0'), ('T0', 'R0'), ('G0', 'R0'), ('D1', 'R0'), ('G1', 'R0'),
        ('R0', 'D0'), ('C0', 'D1')
    ]
    I.add_edges_from(edges_i)

    results = nc.calc_transm_subst_betweenness(A, B, I)
    expected = {'D0': 0.0, 'D1': 0.0, 'T0': 1.0, 'G0': 0.0, 'G1': 0.0, 'R0': 0.0, 'C0': 0.0}
    assert results == expected

    del A
    A = nx.Graph()
    nodes_a = [
        ('D0', {'role': 'distribution_substation'}),
        ('D1', {'role': 'distribution_substation'}),
        ('T0', {'role': 'transmission_substation'}),
        ('T1', {'role': 'transmission_substation'}),
        ('G0', {'role': 'generator'}),
        ('G1', {'role': 'generator'})
    ]
    A.add_nodes_from(nodes_a)
    edges_a = [('D0', 'T0'), ('T0', 'G0'), ('T0', 'T1'), ('D1', 'T1'), ('T1', 'G1')]
    A.add_edges_from(edges_a)

    I.add_edges_from([('T1', 'C0'), ('T1', 'R0')])

    results = nc.calc_transm_subst_betweenness(A, B, I)
    expected = {'D0': 0.0, 'D1': 0.0, 'T0': 3./4, 'T1': 3./4, 'G0': 0.0, 'G1': 0.0, 'R0': 0.0, 'C0': 0.0}
    assert results == expected

    A.remove_edges_from(A.edges())
    edges_a = [('D0', 'T0'), ('T0', 'G0'), ('T0', 'G1'), ('D1', 'T1'), ('T1', 'G1')]
    A.add_edges_from(edges_a)

    # I don't particularly like that there is a path passing through a generator, but it's allowed
    results = nc.calc_transm_subst_betweenness(A, B, I)
    expected = {'D0': 0.0, 'D1': 0.0, 'T0': 3./4, 'T1': 2./4, 'G0': 0.0, 'G1': 0.0, 'R0': 0.0, 'C0': 0.0}
    assert results == expected

    A.remove_edges_from(A.edges())
    edges_a = [('D0', 'T0'), ('T0', 'G0'), ('G0', 'G1'), ('D1', 'T1'), ('T1', 'G1')]
    A.add_edges_from(edges_a)

    del B
    B = nx.Graph()
    nodes_b = [
        ('C0', {'role': 'controller'}),
        ('R0', {'role': 'relay'}),
        ('R1', {'role': 'relay'})
    ]
    B.add_nodes_from(nodes_b)
    edges_b = [('R0', 'C0'), ('R1', 'C0')]
    B.add_edges_from(edges_b)

    I.add_edge('R1', 'D0')

    results = nc.calc_transm_subst_betweenness(A, B, I)
    expected = {'D0': 0.0, 'D1': 0.0, 'T0': 4./6, 'T1': 2./6, 'G0': 0.0, 'G1': 0.0, 'R0': 0.0, 'R1': 0.0, 'C0': 0.0}
    assert results == expected


def test_is_graph_equal():
    # directed vs undirected
    G1 = nx.Graph()
    G2 = nx.DiGraph()
    assert sf.is_graph_equal(G1, G2) is False

    # simple graph vs multigraph
    G2 = nx.MultiGraph()
    assert sf.is_graph_equal(G1, G2) is False

    G2 = nx.Graph()
    assert sf.is_graph_equal(G1, G2) is True

    # different names
    G1.name = 'G'
    G2.name = 'G2'
    assert sf.is_graph_equal(G1, G2) is False

    G2.name = 'G'
    assert sf.is_graph_equal(G1, G2) is True

    # different graph data
    G1.graph['color'] = 'black'
    G2.graph['color'] = 'white'
    assert sf.is_graph_equal(G1, G2, True) is False
    assert sf.is_graph_equal(G1, G2, False) is True

    G2.graph['color'] = 'black'
    assert sf.is_graph_equal(G1, G2, True) is True

    # different number of nodes
    G1.add_nodes_from([0, 1, 2])
    assert sf.is_graph_equal(G1, G2) is False

    G2.add_nodes_from([0, 1, 2])
    assert sf.is_graph_equal(G1, G2) is True

    # different nodes
    G1.add_node(3)
    G2.add_node(4)
    assert sf.is_graph_equal(G1, G2) is False

    G1.add_node(4)
    G2.add_node(3)
    assert sf.is_graph_equal(G1, G2) is True

    # different node data
    G1.node[0]['color'] = 'black'
    G2.node[0]['color'] = 'white'
    assert sf.is_graph_equal(G1, G2, True) is False
    assert sf.is_graph_equal(G1, G2, False) is True

    G2.node[0]['color'] = 'black'
    assert sf.is_graph_equal(G1, G2, True) is True

    # different edge count
    G1.add_edges_from([(0, 1), (1, 2)])
    G2.add_edge(0, 1)
    assert sf.is_graph_equal(G1, G2) is False

    G2.add_edge(1, 2)
    assert sf.is_graph_equal(G1, G2) is True

    # different edges
    G1.add_edge(2, 3)
    G2.add_edge(3, 4)
    assert sf.is_graph_equal(G1, G2) is False

    G1.add_edge(3, 4)
    G2.add_edge(2, 3)
    assert sf.is_graph_equal(G1, G2) is True

    # different edges (loops)
    G1.add_edge(0, 0)
    G2.add_edge(4, 4)
    assert sf.is_graph_equal(G1, G2) is False

    G1.add_edge(4, 4)
    G2.add_edge(0, 0)
    assert sf.is_graph_equal(G1, G2) is True

    # reset
    G1 = nx.Graph()
    G2 = nx.Graph()

    # try to trick it by adding nodes in a different order
    G1.add_edges_from([(2, 1), (0, 1)])
    G2.add_edges_from([(0, 1), (1, 2)])
    assert sf.is_graph_equal(G1, G2) is True

    # different edge data
    G1.edge[0][1]['color'] = 'black'
    G2.edge[0][1]['color'] = 'white'
    assert sf.is_graph_equal(G1, G2, True) is False

    G2.edge[0][1]['color'] = 'black'
    assert sf.is_graph_equal(G1, G2, True) is True

    # reset to directed graphs
    G1 = nx.DiGraph()
    G2 = nx.DiGraph()

    # different edges
    G1.add_edge(0, 1)
    G2.add_edge(1, 0)
    assert sf.is_graph_equal(G1, G2) is False

    G1.add_edge(1, 0)
    G2.add_edge(0, 1)
    assert sf.is_graph_equal(G1, G2) is True

    # different edge data
    G1.edge[0][1]['color'] = 'black'
    G2.edge[0][1]['color'] = 'white'
    assert sf.is_graph_equal(G1, G2, True) is False

    G2.edge[0][1]['color'] = 'black'
    assert sf.is_graph_equal(G1, G2, True) is True

    G1.edge[1][0]['color'] = 'black'
    assert sf.is_graph_equal(G1, G2, True) is False

    G2.edge[1][0]['color'] = 'white'
    assert sf.is_graph_equal(G1, G2, True) is False

    G2.edge[1][0]['color'] = 'black'
    assert sf.is_graph_equal(G1, G2, True) is True


def test_graph_diff():
    # directed vs undirected
    G1 = nx.Graph()
    G2 = nx.DiGraph()
    diff = sf.graph_diff(G1, G2)
    assert diff != ''

    # simple graph vs multigraph
    G2 = nx.MultiGraph()
    diff = sf.graph_diff(G1, G2)
    assert diff != ''

    G2 = nx.Graph()
    assert sf.graph_diff(G1, G2) == ''

    # different names
    G1.name = 'G'
    G2.name = 'G2'
    diff = sf.graph_diff(G1, G2)
    assert diff != ''

    G2.name = 'G'
    assert sf.graph_diff(G1, G2) == ''

    # different graph data
    G1.graph['color'] = 'black'
    G2.graph['color'] = 'white'
    diff = sf.graph_diff(G1, G2, True)
    assert diff != ''

    assert sf.graph_diff(G1, G2, False) == ''

    G2.graph['color'] = 'black'
    assert sf.graph_diff(G1, G2, True) == ''

    # different number of nodes
    G1.add_nodes_from([0, 1, 2])
    diff = sf.graph_diff(G1, G2)
    assert diff != ''

    G2.add_nodes_from([0, 1, 2])
    assert sf.graph_diff(G1, G2) == ''

    # different nodes
    G1.add_node(3)
    G2.add_node(4)
    diff = sf.graph_diff(G1, G2)
    assert diff != ''

    G1.add_node(4)
    G2.add_node(3)
    assert sf.graph_diff(G1, G2) == ''

    # different node data
    G1.node[0]['color'] = 'black'
    G2.node[0]['color'] = 'white'
    diff = sf.graph_diff(G1, G2, True)
    assert diff != ''

    assert sf.graph_diff(G1, G2, False) == ''

    G2.node[0]['color'] = 'black'
    assert sf.graph_diff(G1, G2, True) == ''

    # different edge count
    G1.add_edges_from([(0, 1), (1, 2)])
    G2.add_edge(0, 1)
    diff = (G1, G2)
    assert diff != ''

    G2.add_edge(1, 2)
    assert sf.graph_diff(G1, G2) == ''

    # different edges
    G1.add_edge(2, 3)
    G2.add_edge(3, 4)
    diff = (G1, G2)
    assert diff != ''

    G1.add_edge(3, 4)
    G2.add_edge(2, 3)
    assert sf.graph_diff(G1, G2) == ''

    # different edges (loops)
    G1.add_edge(0, 0)
    G2.add_edge(4, 4)
    diff = (G1, G2)
    assert diff != ''

    G1.add_edge(4, 4)
    G2.add_edge(0, 0)
    assert sf.graph_diff(G1, G2) == ''

    # reset
    G1 = nx.Graph()
    G2 = nx.Graph()

    # try to trick it by adding nodes in a different order
    G1.add_edges_from([(2, 1), (0, 1)])
    G2.add_edges_from([(0, 1), (1, 2)])
    assert sf.graph_diff(G1, G2) == ''

    # different edge data
    G1.edge[0][1]['color'] = 'black'
    G2.edge[0][1]['color'] = 'white'
    diff = (G1, G2, True)
    assert diff != ''

    G2.edge[0][1]['color'] = 'black'
    assert sf.graph_diff(G1, G2, True) == ''

    # reset to directed graphs
    G1 = nx.DiGraph()
    G2 = nx.DiGraph()

    # different edges
    G1.add_edge(0, 1)
    G2.add_edge(1, 0)
    diff = (G1, G2)
    assert diff != ''

    G1.add_edge(1, 0)
    G2.add_edge(0, 1)
    assert sf.graph_diff(G1, G2) == ''

    # different edge data
    G1.edge[0][1]['color'] = 'black'
    G2.edge[0][1]['color'] = 'white'
    diff = (G1, G2, True)
    assert diff != ''

    G2.edge[0][1]['color'] = 'black'
    assert sf.graph_diff(G1, G2, True) == ''

    G1.edge[1][0]['color'] = 'black'
    diff = (G1, G2, True)
    assert diff != ''

    G2.edge[1][0]['color'] = 'white'
    diff = (G1, G2, True)
    assert diff != ''

    G2.edge[1][0]['color'] = 'black'
    assert sf.graph_diff(G1, G2, True) == ''


def test_compare_roles_by_pos():
    G1 = nx.Graph()
    G2 = nx.Graph()

    # different number of nodes, no role diff
    G1.add_node(0, {'x': 0, 'y': 0, 'role': 'black'})
    assert sf.compare_roles_by_pos(G1, G2) == ''

    G2.add_node(0, {'x': 0, 'y': 0, 'role': 'black'})
    assert sf.compare_roles_by_pos(G1, G2) == ''

    # different node ids, same roles, no role diff
    G1.add_node(1, {'x': 1, 'y': 1, 'role': 'black'})
    G2.add_node(1, {'x': 1, 'y': 1, 'role': 'white'})
    diff = sf.compare_roles_by_pos(G1, G2)
    assert diff != ''

    G2.node[1]['role'] = 'black'
    assert sf.compare_roles_by_pos(G1, G2) == ''


def test_compare_links_between_pos():
    # warning for comparing directed and undirected graphs
    G1 = nx.Graph()
    G2 = nx.DiGraph()
    diff = sf.compare_links_between_pos(G1, G2)
    assert diff != ''

    # empty undirected graphs
    G2 = nx.Graph()
    assert sf.compare_links_between_pos(G1, G2) == ''

    # different number of nodes, no edges, no difference
    G1.add_node(0, {'x': 0, 'y': 0})
    assert sf.compare_links_between_pos(G1, G2) == ''

    G2.add_node(0, {'x': 0, 'y': 0})
    assert sf.compare_links_between_pos(G1, G2) == ''

    # different nodes, no edges, no difference
    G1.add_node(1, {'x': 1, 'y': 1})
    G2.add_node(2, {'x': 2, 'y': 2})
    assert sf.compare_links_between_pos(G1, G2) == ''

    G1.add_node(2, {'x': 2, 'y': 2})
    G2.add_node(1, {'x': 1, 'y': 1})
    assert sf.compare_links_between_pos(G1, G2) == ''

    # different edge count
    G1.add_edge(0, 1)
    diff = sf.compare_links_between_pos(G1, G2)
    assert diff != ''

    G2.add_edge(0, 1)
    assert sf.compare_links_between_pos(G1, G2) == ''

    G1.add_node(3, {'x': 3, 'y': 3})
    G2.add_node(3, {'x': 3, 'y': 3})

    # different sets of undirected edges
    G1.add_edge(1, 2)
    G2.add_edge(2, 3)
    diff = sf.compare_links_between_pos(G1, G2)
    assert diff != ''

    G1.add_edge(2, 3)
    G2.add_edge(1, 2)
    assert sf.compare_links_between_pos(G1, G2) == ''

    # undirected loop
    G1.add_edge(0, 0)
    diff = sf.compare_links_between_pos(G1, G2)
    assert diff != ''

    G2.add_edge(0, 0)
    assert sf.compare_links_between_pos(G1, G2) == ''

    # another couple of undirected graphs
    G1 = nx.Graph()
    G2 = nx.Graph()

    # different node ids, same node positions, no edges so no difference
    G1.add_node('0a', {'x': 0, 'y': 0})
    G1.add_node('1a', {'x': 1, 'y': 1})
    G2.add_node('0b', {'x': 0, 'y': 0})
    G2.add_node('1b', {'x': 1, 'y': 1})
    assert sf.compare_links_between_pos(G1, G2) == ''

    # different edge count
    G1.add_edge('0a', '1a')
    diff = sf.compare_links_between_pos(G1, G2)
    assert diff != ''

    G2.add_edge('0b', '1b')
    assert sf.compare_links_between_pos(G1, G2) == ''

    G1.add_node('2a', {'x': 2, 'y': 2})
    G1.add_node('3a', {'x': 3, 'y': 3})
    G2.add_node('2b', {'x': 2, 'y': 2})
    G2.add_node('3b', {'x': 3, 'y': 3})

    # different sets of undirected edges
    G1.add_edge('1a', '2a')
    G2.add_edge('2b', '3b')
    diff = sf.compare_links_between_pos(G1, G2)
    assert diff != ''

    G1.add_edge('2a', '3a')
    G2.add_edge('1b', '2b')
    assert sf.compare_links_between_pos(G1, G2) == ''

    # directed graphs
    G1 = nx.DiGraph()
    G2 = nx.DiGraph()

    # same node ids but with different positions, no edges so no difference
    G1.add_node('a', {'x': 1, 'y': 1})
    G1.add_node('b', {'x': -1, 'y': -1})
    G2.add_node('a', {'x': -1, 'y': -1})
    G2.add_node('b', {'x': 1, 'y': 1})
    assert sf.compare_links_between_pos(G1, G2) == ''

    # different edge count
    G1.add_edge('a', 'b')  # (1, 1) > (-1, -1)
    diff = sf.compare_links_between_pos(G1, G2)
    assert diff != ''

    G2.add_edge('b', 'a')  # (1, 1) > (-1, -1)
    assert sf.compare_links_between_pos(G1, G2) == ''

    G1.add_node('c', {'x': 2, 'y': 2})
    G1.add_node('d', {'x': -2, 'y': -2})
    G2.add_node('c', {'x': -2, 'y': -2})
    G2.add_node('d', {'x': 2, 'y': 2})

    # different sets of directed edges
    G1.add_edge('b', 'c')  # (-1, -1) > (2, 2)
    G2.add_edge('c', 'd')  # (-2, -2) > (2, 2)
    diff = sf.compare_links_between_pos(G1, G2)
    assert diff != ''

    G1.add_edge('d', 'c')  # (-2, -2) > (2, 2)
    G2.add_edge('a', 'd')  # (-1, -1) > (2, 2)
    assert sf.compare_links_between_pos(G1, G2) == ''

    # another couple of directed graphs
    G1 = nx.DiGraph()
    G2 = nx.DiGraph()

    G1.add_node(0, {'x': 0, 'y': 0})
    G1.add_node(1, {'x': 1, 'y': 1})
    G2.add_node(0, {'x': 0, 'y': 0})
    G2.add_node(1, {'x': 1, 'y': 1})

    # directed loop
    G1.add_edge(0, 0)
    diff = sf.compare_links_between_pos(G1, G2)
    assert diff != ''

    G2.add_edge(0, 0)
    assert sf.compare_links_between_pos(G1, G2) == ''

    # different edge data
    G1.add_edge(0, 1, {'color': 'white'})
    G2.add_edge(0, 1, {'color': 'black'})
    assert sf.compare_links_between_pos(G1, G2, data=False) == ''

    diff = sf.compare_links_between_pos(G1, G2, data=True)
    assert diff != ''

    G2.edge[0][1]['color'] = 'white'
    assert sf.compare_links_between_pos(G1, G2, data=True) == ''

    # same edge data
    G1.add_node(2, {'x': 2, 'y': 2})
    G2.add_node(2, {'x': 2, 'y': 2})
    G1.add_edge(1, 2, {'color': 'black'})
    G2.add_edge(1, 2, {'color': 'black'})
    assert sf.compare_links_between_pos(G1, G2, data=False) == ''
    assert sf.compare_links_between_pos(G1, G2, data=True) == ''
