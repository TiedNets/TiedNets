__author__ = 'Agostino Sturaro'

import os
import sys
import random
import logging
import netw_creator as nc
import shared_functions as sf

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0


def write_conf(instance_dir, conf_fpath, a_options, b_options, inter_options, seed):
    config = ConfigParser()

    config.add_section('paths')
    config.set('paths', 'netw_dir', instance_dir)

    config.add_section('build_a')
    for opt_name in a_options:
        config.set('build_a', opt_name, a_options[opt_name])

    config.add_section('build_b')
    for opt_name in b_options:
        config.set('build_b', opt_name, b_options[opt_name])

    config.add_section('build_inter')
    for opt_name in inter_options:
        config.set('build_inter', opt_name, inter_options[opt_name])

    config.add_section('misc')
    config.set('misc', 'seed', seed)

    with open(conf_fpath, 'w') as configfile:
        config.write(configfile)


this_dir = os.path.normpath(os.path.dirname(__file__))
os.chdir(this_dir)
sf.setup_logging('logging_base_conf.json')
logger = logging.getLogger(__name__)
base_dir = os.path.normpath('../Simulations/test_MN')

build_a_options = [{
    'name': 'A',
    'model': 'user_defined_graph',
    'graph_fpath': '../Simulations/MN_data/MN_pow.graphml',
    'file_format': 'graphml',
    'roles': 'random_gen_transm_distr',
    'preassigned_roles_fpath': '../Simulations/MN_data/MN_pow_roles.json',
    'generators': '0',
    'distribution_substations': '714',
    'transmission_substations': '306'
}]

build_b_options = [{
    'name': 'B',
    'model': 'user_defined_graph',
    'graph_fpath': '../Simulations/MN_data/MN_com.graphml',
    'file_format': 'graphml',
    'roles': 'relay_attached_controllers',
    'controllers': '1',
    'relays': '1079'
}]

build_inter_options = [{
    'name': 'Inter',
    'k': '1',
    'dependency_model': 'k-to-n',
    'n': '1089',
    'produce_max_matching': 'True',
    'max_matching_name': 'InterMM'
}]

instances_per_type = 1
seeds = list()
first_group = True

my_random = random.Random(256)

# create directory if it does not exist, clean it if it already exists
sf.makedirs_clean(base_dir, False)

# outer cycle sets different network structure parameters, mixing build options for the 2 networks
instance_num = 0
line_num = 0
for a_opts, b_opts, inter_opts in zip(build_a_options, build_b_options, build_inter_options):

    # inner cycle creates a number of instances with the same structure
    created_for_type = 0
    while created_for_type < instances_per_type:
        if first_group is True:
            seed = my_random.randint(0, sys.maxsize)
            seeds.append(seed)
        else:
            seed = seeds[instance_num % instances_per_type]
            print('seeds[{} % {}] = {}'.format(instance_num, instances_per_type, seed))
        instance_dir = os.path.join(base_dir, 'instance_{}'.format(instance_num))
        conf_fpath = os.path.join(base_dir, 'config_{}.ini'.format(instance_num))
        write_conf(instance_dir, conf_fpath, a_opts, b_opts, inter_opts, seed)
        nc.run(conf_fpath)
        created_for_type += 1
        instance_num += 1

    line_num += 1
    first_group = False
