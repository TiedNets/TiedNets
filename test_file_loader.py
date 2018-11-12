import os
import file_loader as fl

__author__ = 'Agostino Sturaro'

def test_floader():
    global this_dir, logging_conf_fpath
    cache_size = 2
    floader = fl.FileLoader(True, cache_size)
    assert floader.cache_size == cache_size

    missing_file = floader.fetch_json('test_sets/file_loader/missing_file.json')
    assert missing_file is None

    fpath_0 = os.path.abspath('test_sets/file_loader/file_0.json')
    file_0 = floader.fetch_json(fpath_0)
    time_0 = floader.last_hit[fpath_0]
    expected_file_0 = {'name': 'file_0.json'}
    assert file_0 == expected_file_0

    expected_cache = {fpath_0: expected_file_0}
    assert floader.loaded == expected_cache

    fpath_1 = os.path.abspath('test_sets/file_loader/file_1.json')
    file_1 = floader.fetch_json(fpath_1)
    time_1 = floader.last_hit[fpath_1]
    assert time_1 > time_0
    expected_file_1 = {'name': 'file_1.json'}
    assert file_1 == expected_file_1

    expected_cache = {fpath_0: expected_file_0, fpath_1: expected_file_1}
    assert floader.loaded == expected_cache

    fpath_2 = os.path.abspath('test_sets/file_loader/file_2.json')
    file_2 = floader.fetch_json(fpath_2)
    time_2 = floader.last_hit[fpath_2]
    assert time_2 > time_1
    expected_file_2 = {'name': 'file_2.json'}
    assert file_2 == expected_file_2

    # check that the cache was refreshed correctly (keep latest used)
    expected_cache = {fpath_1: expected_file_1, fpath_2: expected_file_2}
    assert floader.loaded == expected_cache

    # make sure FileLoader is returning a copy
    file_2_copy = floader.fetch_json(fpath_2)
    time_3 = floader.last_hit[fpath_2]
    assert time_3 > time_2

    file_2_copy['other'] = 'value'
    assert file_2_copy != file_2
    assert file_2 == expected_file_2

    # test deepcopy algorithm used by FileLoader, G has a single directed edge with color=black
    fpath_3 = os.path.abspath('test_sets/file_loader/file_3.json')
    G = floader.fetch_graphml(fpath_3, str)
    G.edge['a']['b']['color'] = 'white'
    G_copy = floader.fetch_graphml(fpath_3, str)
    expected_edge = ('a', 'b', {'color': 'black'})
    edges = list(G_copy.edges(data=True))
    assert edges[0] == expected_edge
