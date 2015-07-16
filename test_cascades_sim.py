__author__ = 'Agostino Sturaro'

import os
import filecmp
import shutil
from itertools import izip
import cascades_sim as cs
import shared_functions as sf
import networkx as nx

this_dir = os.path.normpath(os.path.dirname(__file__))
logging_conf_fpath = os.path.join(this_dir, 'logging_base_conf.json')


def test_choose_random_nodes():
    # given
    G = nx.Graph()
    G.add_nodes_from(range(0, 11, 1))

    # when
    chosen_nodes = cs.choose_random_nodes(G, 3, seed=128)

    # then
    assert chosen_nodes == [5, 8, 9]


# tests for example 1

def test_run_ex_1_realistic():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_1_full/run_realistic.ini'
    exp_log_fpath = 'test_sets/ex_1_full/exp_log_realistic.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_1_full/res_realistic')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_1.tsv')))


def test_run_ex_1_kngc():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_1_full/run_kngc.ini'
    exp_log_fpath = 'test_sets/ex_1_full/exp_log_kngc.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_1_full/res_kngc')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_1.tsv')))


def test_run_ex_1_sc_th_3():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_1_full/run_sc_th_3.ini'
    exp_log_fpath = 'test_sets/ex_1_full/exp_log_sc_th_3.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_1_full/res_sc_th_3')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_1.tsv')))


def test_run_ex_1_sc_th_4():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_1_full/run_sc_th_4.ini'
    exp_log_fpath = 'test_sets/ex_1_full/exp_log_sc_th_4.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_1_full/res_sc_th_4')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_1.tsv')))


def test_run_ex_1_uniform():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_1_max_matching/run_uniform.ini'
    exp_log_fpath = 'test_sets/ex_1_max_matching/exp_log_uniform.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_1_max_matching/res_uniform')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_1.tsv')))


# tests for example 2a

def test_run_ex_2a_realistic():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2a_full/run_realistic.ini'
    exp_log_fpath = 'test_sets/ex_2a_full/exp_log_realistic.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2a_full/res_realistic')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2a.tsv')))


def test_run_ex_2a_kngc():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2a_full/run_kngc.ini'
    exp_log_fpath = 'test_sets/ex_2a_full/exp_log_kngc.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2a_full/res_kngc')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2a.tsv')))


def test_run_ex_2a_sc_th_3():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2a_full/run_sc_th_3.ini'
    exp_log_fpath = 'test_sets/ex_2a_full/exp_log_sc_th_3.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2a_full/res_sc_th_3')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2a.tsv')))


def test_run_ex_2a_sc_th_4():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2a_full/run_sc_th_4.ini'
    exp_log_fpath = 'test_sets/ex_2a_full/exp_log_sc_th_4.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2a_full/res_sc_th_4')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2a.tsv')))


def test_run_ex_2a_uniform():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2a_max_matching/run_uniform.ini'
    exp_log_fpath = 'test_sets/ex_2a_max_matching/exp_log_uniform.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2a_max_matching/res_uniform')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2a.tsv')))


# tests for example 2b

def test_run_ex_2b_realistic():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2b_full/run_realistic.ini'
    exp_log_fpath = 'test_sets/ex_2b_full/exp_log_realistic.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2b_full/res_realistic')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2b.tsv')))


def test_run_ex_2b_kngc():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2b_full/run_kngc.ini'
    exp_log_fpath = 'test_sets/ex_2b_full/exp_log_kngc.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2b_full/res_kngc')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2b.tsv')))


def test_run_ex_2b_sc_th_3():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2b_full/run_sc_th_3.ini'
    exp_log_fpath = 'test_sets/ex_2b_full/exp_log_sc_th_3.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2b_full/res_sc_th_3')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2b.tsv')))


def test_run_ex_2b_sc_th_4():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2b_full/run_sc_th_4.ini'
    exp_log_fpath = 'test_sets/ex_2b_full/exp_log_sc_th_4.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2b_full/res_sc_th_4')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2b.tsv')))


def test_run_ex_2b_uniform():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2b_max_matching/run_uniform.ini'
    exp_log_fpath = 'test_sets/ex_2b_max_matching/exp_log_uniform.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2b_max_matching/res_uniform')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2b.tsv')))


# tests for example 3

def test_run_ex_3_realistic():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_3_full/run_realistic.ini'
    exp_log_fpath = 'test_sets/ex_3_full/exp_log_realistic.txt'
    exp_end_stats_fpath = os.path.normpath('test_sets/ex_3_full/exp_end_stats_realistic.tsv')
    res_end_stats_fpath = os.path.normpath('test_sets/useless/useless_3.tsv')

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # confront the 2 files line by line, without caring about different line endings
    with open(exp_end_stats_fpath, 'r') as exp_file, open(res_end_stats_fpath, 'r') as res_file:
        lines_match = all(a == b for a, b in izip(exp_file, res_file))
    assert lines_match is True

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_3_full/res_realistic')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_3.tsv')))


# def test_run_ex_3_realistic_with_reasons():
#     # given
#     global this_dir, logging_conf_fpath
#     sim_conf_fpath = 'test_sets/ex_3_full/run_realistic_w_causes.ini'
#     exp_log_fpath = 'test_sets/ex_3_full/exp_log_realistic_w_causes.txt'
#     exp_end_stats_fpath = os.path.normpath('test_sets/ex_3_full/exp_end_stats_realistic_w_causes.tsv')
#     res_end_stats_fpath = os.path.normpath('test_sets/useless/useless_3.tsv')
#
#     # when
#     os.chdir(this_dir)
#     sf.setup_logging(logging_conf_fpath)
#     cs.run(sim_conf_fpath)
#
#     # then
#     assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used
#
#     # confront the 2 files line by line, without caring about different line endings
#     same_end_stats = False
#     if open(exp_end_stats_fpath, 'r').read() == open(res_end_stats_fpath, 'r').read():
#         same_end_stats = True
#     assert same_end_stats is True
#
#     # tear down
#     shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_3_full/res_realistic_w_causes')))
#     # os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_3.tsv')))


def test_run_ex_3_kngc():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_3_full/run_kngc.ini'
    exp_log_fpath = 'test_sets/ex_3_full/exp_log_kngc.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_3_full/res_kngc')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_3.tsv')))


def test_run_ex_3_sc_th_3():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_3_full/run_sc_th_3.ini'
    exp_log_fpath = 'test_sets/ex_3_full/exp_log_sc_th_3.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_3_full/res_sc_th_3')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_3.tsv')))


def test_run_ex_3_sc_th_4():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_3_full/run_sc_th_4.ini'
    exp_log_fpath = 'test_sets/ex_3_full/exp_log_sc_th_4.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_3_full/res_sc_th_4')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_3.tsv')))


def test_run_ex_3_uniform():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_3_max_matching/run_uniform.ini'
    exp_log_fpath = 'test_sets/ex_3_max_matching/exp_log_uniform.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_3_max_matching/res_uniform')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_3.tsv')))


# test instabilities

def test_run_ex_unstable_1():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2a_full/run_sc_th_5.ini'
    exp_log_fpath = 'test_sets/ex_2a_full/exp_log_sc_th_5.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2a_full/res_sc_th_5')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2a.tsv')))


def test_run_ex_unstable_2():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_unstable_2/run_uniform.ini'
    exp_log_fpath = 'test_sets/ex_unstable_2/exp_log_uniform.txt'

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath)

    # then
    assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_unstable_2/res_uniform')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_4.tsv')))


def test_choose_most_inter_used_nodes():
    global this_dir, logging_conf_fpath
    netw_a_fpath = os.path.join(this_dir, os.path.normpath('test_sets/ex_1_full/A.graphml'))
    netw_inter_fpath = os.path.join(this_dir, os.path.normpath('test_sets/ex_1_full/Inter.graphml'))

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    A = nx.read_graphml(netw_a_fpath)
    I = nx.read_graphml(netw_inter_fpath)
    chosen_nodes_1 = cs.choose_most_inter_used_nodes(A, I, 1, 'distribution_substation')
    chosen_nodes_2 = cs.choose_most_inter_used_nodes(A, I, 2, 'distribution_substation')
    chosen_nodes_3 = cs.choose_most_inter_used_nodes(A, I, 3, 'distribution_substation')

    # then
    assert chosen_nodes_1 == ['D2']
    assert sorted(chosen_nodes_2) == ['D2', 'D3']
    assert sorted(chosen_nodes_3) == ['D1', 'D2', 'D3']

def test_choose_most_intra_used_nodes():
    global this_dir, logging_conf_fpath
    netw_a_fpath = os.path.join(this_dir, os.path.normpath('test_sets/ex_1_full/A.graphml'))
    netw_inter_fpath = os.path.join(this_dir, os.path.normpath('test_sets/ex_1_full/Inter.graphml'))

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    A = nx.read_graphml(netw_a_fpath)
    I = nx.read_graphml(netw_inter_fpath)
    chosen_nodes_1 = cs.choose_most_intra_used_nodes(A, 1, 'transmission_substation')
    chosen_nodes_2 = cs.choose_most_intra_used_nodes(A, 2, 'transmission_substation')
    chosen_nodes_3 = cs.choose_most_intra_used_nodes(A, 3, 'transmission_substation')

    # then
    assert chosen_nodes_1 == ['T1']
    assert sorted(chosen_nodes_2) == ['T1', 'T3']
    assert sorted(chosen_nodes_3) == ['T1', 'T2', 'T3']
