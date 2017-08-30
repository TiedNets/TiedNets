import os
import shutil
import file_loader as fl
import cascades_sim as cs
import shared_functions as sf
import networkx as nx

__author__ = 'Agostino Sturaro'

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


def test_choose_nodes_by_rank():
    rank_node_pairs = [(1, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    chosen_nodes = cs.pick_nodes_by_score(rank_node_pairs, 5)
    assert chosen_nodes == [5, 4, 3, 2, 1]

    chosen_nodes = cs.pick_nodes_by_score(rank_node_pairs, 3)
    assert chosen_nodes == [5, 4, 3]

    chosen_nodes = cs.pick_nodes_by_score(rank_node_pairs, 0)
    assert chosen_nodes == []


# tests for example 1

def test_run_ex_1_realistic():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_1_full/run_realistic.ini'
    exp_log_fpath = 'test_sets/ex_1_full/exp_log_realistic.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_1_full/res_realistic')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_1.tsv')))


def test_run_ex_1_caching():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_1_full/run_realistic.ini'
    exp_log_fpath = 'test_sets/ex_1_full/exp_log_realistic.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_1_full/res_realistic')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_1.tsv')))

    # when
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_1_full/res_realistic')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_1.tsv')))


def test_run_ex_1_kngc():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_1_full/run_kngc.ini'
    exp_log_fpath = 'test_sets/ex_1_full/exp_log_kngc.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_1_full/res_kngc')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_1.tsv')))


def test_run_ex_1_sc_th_3():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_1_full/run_sc_th_3.ini'
    exp_log_fpath = 'test_sets/ex_1_full/exp_log_sc_th_3.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_1_full/res_sc_th_3')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_1.tsv')))


def test_run_ex_1_sc_th_4():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_1_full/run_sc_th_4.ini'
    exp_log_fpath = 'test_sets/ex_1_full/exp_log_sc_th_4.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_1_full/res_sc_th_4')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_1.tsv')))


def test_run_ex_1_uniform():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_1_max_matching/run_uniform.ini'
    exp_log_fpath = 'test_sets/ex_1_max_matching/exp_log_uniform.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_1_max_matching/res_uniform')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_1.tsv')))


# tests for example 2a

def test_run_ex_2a_realistic():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2a_full/run_realistic.ini'
    exp_log_fpath = 'test_sets/ex_2a_full/exp_log_realistic.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2a_full/res_realistic')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2a.tsv')))


def test_run_ex_2a_kngc():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2a_full/run_kngc.ini'
    exp_log_fpath = 'test_sets/ex_2a_full/exp_log_kngc.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2a_full/res_kngc')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2a.tsv')))


def test_run_ex_2a_sc_th_3():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2a_full/run_sc_th_3.ini'
    exp_log_fpath = 'test_sets/ex_2a_full/exp_log_sc_th_3.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2a_full/res_sc_th_3')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2a.tsv')))


def test_run_ex_2a_sc_th_4():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2a_full/run_sc_th_4.ini'
    exp_log_fpath = 'test_sets/ex_2a_full/exp_log_sc_th_4.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2a_full/res_sc_th_4')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2a.tsv')))


def test_run_ex_2a_uniform():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2a_max_matching/run_uniform.ini'
    exp_log_fpath = 'test_sets/ex_2a_max_matching/exp_log_uniform.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2a_max_matching/res_uniform')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2a.tsv')))


# tests for example 2b

def test_run_ex_2b_realistic():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2b_full/run_realistic.ini'
    exp_log_fpath = 'test_sets/ex_2b_full/exp_log_realistic.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2b_full/res_realistic')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2b.tsv')))


def test_run_ex_2b_kngc():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2b_full/run_kngc.ini'
    exp_log_fpath = 'test_sets/ex_2b_full/exp_log_kngc.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2b_full/res_kngc')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2b.tsv')))


def test_run_ex_2b_sc_th_3():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2b_full/run_sc_th_3.ini'
    exp_log_fpath = 'test_sets/ex_2b_full/exp_log_sc_th_3.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2b_full/res_sc_th_3')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2b.tsv')))


def test_run_ex_2b_sc_th_4():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2b_full/run_sc_th_4.ini'
    exp_log_fpath = 'test_sets/ex_2b_full/exp_log_sc_th_4.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2b_full/res_sc_th_4')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2b.tsv')))


def test_run_ex_2b_uniform():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2b_max_matching/run_uniform.ini'
    exp_log_fpath = 'test_sets/ex_2b_max_matching/exp_log_uniform.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

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
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)
    assert sf.compare_files_by_line(res_end_stats_fpath, exp_end_stats_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_3_full/res_realistic')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_3.tsv')))


def test_run_ex_3_kngc():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_3_full/run_kngc.ini'
    exp_log_fpath = 'test_sets/ex_3_full/exp_log_kngc.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_3_full/res_kngc')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_3.tsv')))


def test_run_ex_3_sc_th_3():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_3_full/run_sc_th_3.ini'
    exp_log_fpath = 'test_sets/ex_3_full/exp_log_sc_th_3.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_3_full/res_sc_th_3')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_3.tsv')))


def test_run_ex_3_sc_th_4():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_3_full/run_sc_th_4.ini'
    exp_log_fpath = 'test_sets/ex_3_full/exp_log_sc_th_4.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_3_full/res_sc_th_4')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_3.tsv')))


def test_run_ex_3_uniform():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_3_max_matching/run_uniform.ini'
    exp_log_fpath = 'test_sets/ex_3_max_matching/exp_log_uniform.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_3_max_matching/res_uniform')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_3.tsv')))


# test instabilities

def test_run_ex_unstable_1():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_2a_full/run_sc_th_5.ini'
    exp_log_fpath = 'test_sets/ex_2a_full/exp_log_sc_th_5.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

    # tear down
    shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_2a_full/res_sc_th_5')))
    os.remove(os.path.join(this_dir, os.path.normpath('test_sets/useless/useless_2a.tsv')))


def test_run_ex_unstable_2():
    # given
    global this_dir, logging_conf_fpath
    sim_conf_fpath = 'test_sets/ex_unstable_2/run_uniform.ini'
    exp_log_fpath = 'test_sets/ex_unstable_2/exp_log_uniform.txt'
    floader = fl.FileLoader()

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    cs.run(sim_conf_fpath, floader)

    # then
    assert sf.compare_files_by_line('log.txt', exp_log_fpath, False)

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

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    A = nx.read_graphml(netw_a_fpath)
    chosen_nodes_1 = cs.choose_most_intra_used_nodes(A, 1, 'transmission_substation')
    chosen_nodes_2 = cs.choose_most_intra_used_nodes(A, 2, 'transmission_substation')
    chosen_nodes_3 = cs.choose_most_intra_used_nodes(A, 3, 'transmission_substation')

    # then
    assert chosen_nodes_1 == ['T1']
    assert sorted(chosen_nodes_2) == ['T1', 'T3']
    assert sorted(chosen_nodes_3) == ['T1', 'T2', 'T3']


def test_find_uncontrolled_pow_nodes():
    global this_dir, logging_conf_fpath
    netw_a_fpath = os.path.join(this_dir, os.path.normpath('test_sets/ex_4_full/A.graphml'))
    netw_b_fpath = os.path.join(this_dir, os.path.normpath('test_sets/ex_4_full/B.graphml'))
    netw_inter_fpath = os.path.join(this_dir, os.path.normpath('test_sets/ex_4_full/Inter.graphml'))

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    A = nx.read_graphml(netw_a_fpath)
    B = nx.read_graphml(netw_b_fpath)
    I = nx.read_graphml(netw_inter_fpath)

    tmp_B = B.copy()
    tmp_I = I.copy()
    tmp_B.remove_node('R1')
    tmp_I.remove_node('R1')
    found_nodes_1 = cs.find_uncontrolled_pow_nodes(A, tmp_B, tmp_I, True)

    tmp_B = B.copy()
    tmp_I = I.copy()
    tmp_B.remove_node('R4')
    tmp_I.remove_node('R4')
    found_nodes_2 = cs.find_uncontrolled_pow_nodes(A, tmp_B, tmp_I, True)

    tmp_B = B.copy()
    tmp_I = I.copy()
    tmp_B.remove_node('C1')
    tmp_I.remove_node('C1')
    found_nodes_3 = cs.find_uncontrolled_pow_nodes(A, tmp_B, tmp_I, True)
    tmp_B.remove_node('C2')
    tmp_I.remove_node('C2')
    found_nodes_4 = cs.find_uncontrolled_pow_nodes(A, tmp_B, tmp_I, True)

    # then
    assert found_nodes_1.keys() == ['no_sup_relays', 'no_com_path', 'no_sup_ccs']
    assert sorted(found_nodes_1['no_sup_relays']) == ['D1', 'G1', 'T1']
    assert sorted(found_nodes_2['no_com_path']) == ['D2', 'G2', 'T2']
    assert found_nodes_3['no_sup_ccs'] == []
    assert sorted(found_nodes_4['no_sup_ccs']) == ['D1', 'D2', 'G1', 'G2', 'T1', 'T2']


def test_calc_stats_on_centrality():
    global this_dir, logging_conf_fpath
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    file_loader = fl.FileLoader()
    centrality_fpath = os.path.join(this_dir, os.path.normpath('test_sets/ex_centralities/centralities.json'))
    centrality_info = file_loader.fetch_json(centrality_fpath)

    centrality_name = 'centrality_a'
    result_key_suffix = 'centr_a'
    attacked_nodes = ['A1', 'A4', 'A8', 'A9']
    exp_centr_stats = {'p_q_1_centr_a': 0.5, 'p_q_2_centr_a': 0.0, 'p_q_3_centr_a': 0.5, 'p_q_4_centr_a': 0.0,
                       'p_q_5_centr_a': 1.0, 'p_tot_centr_a': 0.4, 'sum_centr_a': 4.0}

    centr_stats = cs.calc_atk_centrality_stats(attacked_nodes, centrality_name, result_key_suffix, centrality_info)
    assert centr_stats == exp_centr_stats

    centrality_name = 'centrality_b'
    result_key_suffix = 'centr_b'
    attacked_nodes = ['A1', 'A4', 'A8', 'A9']
    exp_centr_stats = {'p_q_1_centr_b': 0.5, 'p_q_2_centr_b': 0.0, 'p_q_3_centr_b': 0.5, 'p_q_4_centr_b': 0.0,
                       'p_q_5_centr_b': 1.0, 'p_tot_centr_b': (22.0 / 45.0), 'sum_centr_b': 22.0}

    centr_stats = cs.calc_atk_centrality_stats(attacked_nodes, centrality_name, result_key_suffix, centrality_info)
    assert centr_stats == exp_centr_stats
