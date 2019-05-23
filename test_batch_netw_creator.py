import json
import os
import shared_functions as sf
import networkx as nx
import batch_netw_creator as nc

__author__ = 'Agostino Sturaro'

this_dir = os.path.normpath(os.path.dirname(__file__))
logging_conf_fpath = os.path.join(this_dir, 'logging_base_conf.json')


# the compared files must be in two different folders and have the same name

def compare_graphml_in_dirs(file_dir_1, file_dir_2, fname, check_data, ignored_node_keys):
    G1 = nx.read_graphml(os.path.join(file_dir_1, fname), node_type=str)
    G2 = nx.read_graphml(os.path.join(file_dir_2, fname), node_type=str)
    return sf.is_graph_equal(G1, G2, check_data, ignored_node_keys)


def compare_json_in_dirs(file_dir_1, file_dir_2, fname):
    with open(os.path.join(file_dir_1, fname)) as json_file_1:
        nest_1 = json.load(json_file_1)

    with open(os.path.join(file_dir_2, fname)) as json_file_2:
        nest_2 = json.load(json_file_2)

    # this comparison takes care of nested structures
    return nest_1 == nest_2


def compare_output_graphmls_in_dirs(inst_res_dir, inst_exp_dir, check_data=True, ignored_node_keys=[]):
    assert compare_graphml_in_dirs(inst_res_dir, inst_exp_dir, 'A.graphml', check_data, ignored_node_keys) is True
    assert compare_graphml_in_dirs(inst_res_dir, inst_exp_dir, 'B.graphml', check_data, ignored_node_keys) is True
    assert compare_graphml_in_dirs(inst_res_dir, inst_exp_dir, 'Inter.graphml', check_data, ignored_node_keys) is True
    assert compare_graphml_in_dirs(inst_res_dir, inst_exp_dir, 'InterMM.graphml', check_data, ignored_node_keys) is True
    assert compare_graphml_in_dirs(inst_res_dir, inst_exp_dir, 'UnionAB.graphml', check_data, ignored_node_keys) is True


def compare_output_jsons_in_dirs(inst_res_dir, inst_exp_dir):
    assert compare_json_in_dirs(inst_res_dir, inst_exp_dir, 'node_centrality_A.json') is True
    assert compare_json_in_dirs(inst_res_dir, inst_exp_dir, 'node_centrality_B.json') is True
    assert compare_json_in_dirs(inst_res_dir, inst_exp_dir, 'node_centrality_Inter.json') is True
    assert compare_json_in_dirs(inst_res_dir, inst_exp_dir, 'node_centrality_InterMM.json') is True
    assert compare_json_in_dirs(inst_res_dir, inst_exp_dir, 'node_centrality_UnionAB.json') is True
    assert compare_json_in_dirs(inst_res_dir, inst_exp_dir, 'node_centrality_misc.json') is True


# this test might take more than 5 minutes to run
def test_run_synth_1():
    # given
    global this_dir, logging_conf_fpath
    base_dir = os.path.normpath('test_sets/batch_nc_synth_1')
    batch_conf_fpath = os.path.join(base_dir, 'input', 'conf.json')

    expected_output_dir_0 = os.path.join(base_dir, 'expected_output', 'instance_0')
    actual_output_dir_0 = os.path.join(base_dir, 'actual_output', 'instance_0')

    expected_output_dir_1 = os.path.join(base_dir, 'expected_output', 'instance_1')
    actual_output_dir_1 = os.path.join(base_dir, 'actual_output', 'instance_1')

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    nc.run(batch_conf_fpath)

    compare_output_graphmls_in_dirs(actual_output_dir_0, expected_output_dir_0)
    compare_output_jsons_in_dirs(actual_output_dir_0, expected_output_dir_0)

    compare_output_graphmls_in_dirs(actual_output_dir_1, expected_output_dir_1)
    compare_output_jsons_in_dirs(actual_output_dir_1, expected_output_dir_1)

    # tear down
    # sf.ensure_dir_clean(os.path.join(base_dir, 'actual_output'), True, False)


def test_run_real_1():
    # given
    global this_dir, logging_conf_fpath
    base_dir = os.path.normpath('test_sets/batch_nc_real_1')
    batch_conf_fpath = os.path.join(base_dir, 'input/conf.json')

    expected_output_dir_0 = os.path.join(base_dir, 'expected_output/instance_0')
    actual_output_dir_0 = os.path.join(base_dir, 'actual_output/instance_0')

    expected_output_dir_1 = os.path.join(base_dir, 'expected_output/instance_1')
    actual_output_dir_1 = os.path.join(base_dir, 'actual_output/instance_1')

    # when
    os.chdir(this_dir)
    sf.setup_logging(logging_conf_fpath)
    # nc.run(batch_conf_fpath)

    compare_output_graphmls_in_dirs(actual_output_dir_0, expected_output_dir_0)
    compare_output_jsons_in_dirs(actual_output_dir_0, expected_output_dir_0)

    compare_output_graphmls_in_dirs(actual_output_dir_1, expected_output_dir_1)
    compare_output_jsons_in_dirs(actual_output_dir_1, expected_output_dir_1)

    # tear down
    # sf.ensure_dir_clean(res_dir, True, False)