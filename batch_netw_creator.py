import os
import logging
import netw_creator as nc
import shared_functions as sf

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0

__author__ = 'Agostino Sturaro'


def write_conf(instance_dir, conf_fpath, a_options, b_options, inter_options, misc_options):
    config = ConfigParser()

    config.add_section('paths')
    config.set('paths', 'netw_dir', instance_dir)

    config.add_section('build_a')
    for opt_name in a_options:
        config.set('build_a', opt_name, str(a_options[opt_name]))

    config.add_section('build_b')
    for opt_name in b_options:
        config.set('build_b', opt_name, str(b_options[opt_name]))

    config.add_section('build_inter')
    for opt_name in inter_options:
        config.set('build_inter', opt_name, str(inter_options[opt_name]))

    config.add_section('misc')
    for opt_name in misc_options:
        config.set('misc', opt_name, str(misc_options[opt_name]))

    with open(conf_fpath, 'w') as configfile:
        config.write(configfile)


this_dir = os.path.normpath(os.path.dirname(__file__))
os.chdir(this_dir)
sf.setup_logging('logging_base_conf.json')
logger = logging.getLogger(__name__)
base_dir = os.path.normpath('../Simulations/test_mp/MN')

# remember to increase d_0 for bigger networks
build_a_options = [{
    # 'name': 'A',
    # 'model': 'rt_nested_smallworld',
    # 'nodes': 2000,
    # 'subnets': 40,
    # 'beta': 0.2,
    # 'alpha': 0.2,
    # 'd_0': 14,
    # 'avg_k': 4,
    # 'q_rw': 0.5,
    # 'roles': 'subnet_gen_transm_distr',
    # 'generators': 200,
    # 'transmission_substations': 270,
    # 'distribution_substations': 1260
    #
    # 'name': 'A',
    # 'model': 'rt_nested_smallworld',
    # 'nodes': 1000,
    # 'subnets': 20,
    # 'beta': 0.2,
    # 'alpha': 0.2,
    # 'd_0': 7,
    # 'avg_k': 4,
    # 'q_rw': 0.5,
    # 'roles': 'subnet_gen_transm_distr',
    # 'generators': 100,
    # 'transmission_substations': 270,
    # 'distribution_substations': 630
    #
    # 'name': 'A',
    # 'model': 'rt_nested_smallworld',
    # 'nodes': 500,
    # 'subnets': 20,
    # 'beta': 0.2,
    # 'alpha': 0.2,
    # 'd_0': 7,
    # 'avg_k': 4,
    # 'q_rw': 0.5,
    # 'roles': 'subnet_gen_transm_distr',
    # 'generators': 50,
    # 'transmission_substations': 135,
    # 'distribution_substations': 315,
    # 'layout': 'spring',
    # 'controller_attachment': 'random',
    # 'controller_placement': 'random'
    #
    'name': 'A',
    'model': 'user_defined_graph',
    'user_graph_fpath': '../Simulations/MN_data_new/MN_pow.graphml',
    'roles': 'random_gen_transm_distr',
    'preassigned_roles_fpath': '',  # placeholder value, remember to fill
    'generators': 0,
    'distribution_substations': 0,
    'transmission_substations': 0
}]

build_b_options = [{
    # 'name': 'B',
    # 'model': 'barabasi_albert',
    # 'm': 3,
    # 'roles': 'relay_attached_controllers',
    # 'controllers': 1,
    # 'relays': 1999
    #
    # 'name': 'B',
    # 'model': 'barabasi_albert',
    # 'm': 3,
    # 'roles': 'relay_attached_controllers',
    # 'controller_attachment': 'random',
    # 'controllers': 1,
    # 'relays': 999
    #
    # 'name': 'B',
    # 'model': 'barabasi_albert',
    # 'm': 3,
    # 'roles': 'relay_attached_controllers',
    # 'controllers': 1,
    # 'relays': 499,
    # 'layout': 'spring',
    # 'controller_attachment': 'random',
    # 'controller_placement': 'random'
    #
    'name': 'B',
    'model': 'user_defined_graph',
    'user_graph_fpath': '../Simulations/MN_data_new/MN_com.graphml',
    'roles': 'relay_attached_controllers',
    'controllers': 1,
    'controller_placement': 'center',
    'controller_attachment': 'prefer_nearest'
}]

build_inter_options = [{
    # 'name': 'Inter',
    # 'dependency_model': 'k-to-n',
    # 'k': 1,
    # 'n': 2000,
    # 'com_access_points': 1,
    # 'prefer_nearest': False,
    # 'produce_max_matching': True,
    # 'max_matching_name': 'InterMM'
    #
    # 'name': 'Inter',
    # 'dependency_model': 'k-to-n',
    # 'k': 1,
    # 'n': 1000,
    # 'com_access_points': 1,
    # 'prefer_nearest': False,
    # 'produce_max_matching': True,
    # 'max_matching_name': 'InterMM'
    #
    # 'name': 'Inter',
    # 'dependency_model': 'k-to-n',
    # 'k': 1,
    # 'n': 500,
    # 'com_access_points': 1,
    # 'prefer_nearest': False,
    # 'produce_max_matching': True,
    # 'max_matching_name': 'InterMM'
    #
    'name': 'Inter',
    'dependency_model': 'k-to-n',
    'k': 1,
    'n': 1091,
    'com_access_points': 1,
    'prefer_nearest': True,  # geographical attachment
    'produce_max_matching': True,
    'max_matching_name': 'InterMM'
}]

misc_options = [{
    'produce_ab_union': True,
    'ab_union_name': 'UnionAB',
    'calc_node_centrality': True,
    'calc_misc_centralities': True
}]

instances_per_type = 60
first_seed = 0

# in older versions this list was generated randomly, now it's a simple range
seeds = range(first_seed, first_seed + instances_per_type)
print('seeds = {}'.format(seeds))

# create directory if it does not exist, clean it if it already exists
# TODO: this is not asking for confirmation!
sf.makedirs_clean(base_dir, True)

# outer cycle sets different network structure parameters, mixing build options for the 2 networks
instance_num = 0
first_group = True
# TODO: simplify this and take a json without zipping
for a_opts, b_opts, inter_opts, misc_opts in zip(build_a_options, build_b_options, build_inter_options, misc_options):

    # inner cycle creates a number of instances with the same structure
    created_for_type = 0
    while created_for_type < instances_per_type:
        a_opts['preassigned_roles_fpath'] = '../Simulations/MN_data_new/MN_pow_roles_{}.json'.format(instance_num)
        if first_group is True:
            seed = seeds[created_for_type]
        else:
            seed = seeds[instance_num % instances_per_type]
        print('Current seed: {}'.format(seed))
        misc_opts['seed'] = seed
        instance_dir = os.path.join(base_dir, 'instance_{}'.format(instance_num))
        conf_fpath = os.path.join(base_dir, 'config_{}.ini'.format(instance_num))
        write_conf(instance_dir, conf_fpath, a_opts, b_opts, inter_opts, misc_opts)
        nc.run(conf_fpath)
        created_for_type += 1
        instance_num += 1

    first_group = False
