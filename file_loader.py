import os
import json
import time
import copy
import networkx as nx

__author__ = 'Agostino'


class FileLoader:
    def __init__(self, return_copy=True, cache_size=10):
        self.loaded = {}
        self.last_hit = {}
        self.return_copy = return_copy
        self.cache_size = cache_size

    def read_graphml(self, fpath, node_type):
        fpath = os.path.abspath(fpath)
        if not os.path.isfile(fpath):
            return None
        if fpath in self.loaded:
            graph = self.loaded[fpath]
            # print('found fpath {}'.format(fpath))  # debug
        else:
            if len(self.loaded) > self.cache_size:
                stalest = min(self.last_hit.iterkeys(), key=(lambda key: self.last_hit[key]))
                # print('replacing stalest {}'.format(stalest))  # debug
                del self.loaded[stalest]
                del self.last_hit[stalest]
            # print('loading fpath {}'.format(fpath))  # debug
            graph = nx.read_graphml(fpath, node_type=node_type)
            self.loaded[fpath] = graph
        self.last_hit[fpath] = time.clock()
        if self.return_copy:
            graph = copy.deepcopy(graph)
        return graph

    # TODO: share code with previous function
    def read_json(self, fpath, **kwargs):
        fpath = os.path.abspath(fpath)
        if not os.path.isfile(fpath):
            return None
        if fpath in self.loaded:
            json_dict = self.loaded[fpath]
            print('found fpath {}'.format(fpath))  # debug
        else:
            if len(self.loaded) > self.cache_size:
                stalest = min(self.last_hit.iterkeys(), key=(lambda key: self.last_hit[key]))
                print('replacing stalest {}'.format(stalest))  # debug
                del self.loaded[stalest]
                del self.last_hit[stalest]
            with open(fpath, 'r') as json_file:
                print('loading fpath {}'.format(fpath))  # debug
                json_dict = json.load(json_file, **kwargs)
                self.loaded[fpath] = json_dict
        self.last_hit[fpath] = time.clock()
        if self.return_copy:
            json_dict = copy.deepcopy(json_dict)
        return json_dict
