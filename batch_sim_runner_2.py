import os
import csv
import logging
import random
import file_loader as fl
import shared_functions as sf
import cascades_sim as sim

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0

__author__ = 'Agostino Sturaro'


def write_conf(conf_fpath, paths, run_options, misc):
    config = ConfigParser()

    config.add_section('paths')
    for opt_name in paths:
        config.set('paths', opt_name, paths[opt_name])

    config.add_section('run_opts')
    for opt_name in run_options:
        config.set('run_opts', opt_name, run_options[opt_name])

    config.add_section('misc')
    for opt_name in misc:
        config.set('misc', opt_name, misc[opt_name])

    sf.ensure_dir_exists(os.path.dirname(os.path.realpath(conf_fpath)))

    with open(conf_fpath, 'w') as conf_file:
        config.write(conf_file)


# An "instance" is a set of graphs forming an interdependent network (power, telecom and inter), indicated by a number
# A "group" of simulations (sim_group) is intended to gather simulations executed using similar parameters. For example,
# a group can contain simulations on the same instances executed changing only the random seed and the value of another
# variable. In fact, results of the same sim_group are collected in a way that makes it easy to study the effects of
# changing a single variable (the independent variable).
# Because of this, results from a group of simulations are simple to turn into a 2D plot. The easiest case is if a
# single simulation was executed for each value of the independent variable.
# However, the group may contain n simulations for each value of the independent variable if:
# - a simulation was executed for each of n instances
# - a simulation was executed n times on the same instance (e.g. to try different seeds)
# - a simulation was executed m times on q instances (e.g. trying n seeds on each instance), and m*q=n
# In each of these cases, results regarding the same value of the independent variable must be averaged before plotting.

this_dir = os.path.normpath(os.path.dirname(__file__))
os.chdir(this_dir)
sf.setup_logging('logging_base_conf.json')
logger = logging.getLogger(__name__)

# begin of user defined variables
index_fname = '_index.tsv'
instances_dir = '../Simulations/centrality/1cc_1ap'  # parent directory of the instances directories

base_configs = [{
    'paths': {
        'netw_dir': None,  # to be filled by the algorithm
        'netw_b_fname': 'B.graphml',
        'netw_a_fname': 'A.graphml',
        'netw_inter_fname': 'Inter.graphml',
        'netw_union_fname': 'UnionAB.graphml',
        'results_dir': '../Simulations/centrality/1cc_1ap/random/realistic',  # group_results_dir
        'run_stats_fname': None,
        'end_stats_fpath': None,
        'ml_stats_fpath': '../Simulations/centrality/1cc_1ap/random/realistic/ml_stats_0.tsv'
    },

    'run_opts': {
        'attacked_netw': 'Both',
        'attack_tactic': 'random',
        'calc_centrality_on': 'netw_inter',
        'intra_support_type': 'realistic',
        'inter_support_type': 'realistic',
        'save_death_cause': True,
        'save_attacked_roles': True,
        'attacks': None,
        'seed': None
    },

    'misc': {
        'instance': None,
        'sim_group': None
    }
}]

first_instance = 0  # usually 0, unless you want to skip a group, then it should be divisible by sim_group_size
last_instance = 5  # exclusive, should be divisible by sim_group_size

# This script is used to run multiple simulations on a set of instances. At its core is a loop that generates a
# configuration file with a different combination of parameters and executes a simulation based on it.
# If we want to use a group of simulations to draw a 2D plot, showing the behavior obtained by changing a variable, then
# we need to specify the name of the independent variable and the values we want it to assume.

indep_var_name = 'attacks'  # name of the independent variable of the simulation
# indep_var_vals = list(range(0, 3, 1))  # values of the independent value of the simulation
indep_var_vals = sorted(random.sample(range(1, 200), 50))
print('indep_var_vals = {}'.format(indep_var_vals))
# indep_var_vals = list(range(0, 61, 5)) + [69]  # values of the independent value of the simulation
# indep_var_name = 'min_rank'  # name of the independent variable of the simulation, in the run_opts section
# indep_var_vals = list(range(0, 2000, 1))  # values of the independent variable of the simulation

seeds = list(range(100, 120, 1))  # used to execute multiple tests on the same network instance
# seeds = [1]
# end of user defined variables

floader = fl.FileLoader()

for sim_group in range(0, len(base_configs)):
    paths = base_configs[sim_group]['paths']
    run_options = base_configs[sim_group]['run_opts']
    group_results_dir = paths['results_dir']
    sf.ensure_dir_exists(group_results_dir)
    misc = base_configs[sim_group]['misc']
    misc['sim_group'] = sim_group
    runs_by_instance = [0] * (last_instance - first_instance)  # number of simulations executed for each instance

    # group_index will be the index of the config files for each simulation of that group
    group_index_fpath = os.path.join(group_results_dir, 'sim_group_{}_index.tsv'.format(sim_group))
    with open(group_index_fpath, 'wb') as group_index_file:
        group_index = csv.writer(group_index_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        group_index.writerow(['instance', 'instance_conf_fpath'])

    # outer cycle ranging over values of the independent variable
    for var_value in indep_var_vals:
        run_options[indep_var_name] = var_value
        paths['end_stats_fpath'] = os.path.join(group_results_dir, 'sim_group_{}_stats.tsv'.format(sim_group))

        # inner cycle ranging over different network instances
        for instance_num in range(first_instance, last_instance, 1):
            misc['instance'] = instance_num
            paths['netw_dir'] = os.path.join(instances_dir, 'instance_{}'.format(instance_num))  # input

            # inner cycle ranging over different seeds
            for seed in seeds:
                run_options['seed'] = seed
                run_num = runs_by_instance[instance_num]
                paths['results_dir'] = os.path.join(group_results_dir, 'instance_' + str(instance_num),
                                                    'run_' + str(run_num))
                paths['run_stats_fname'] = 'run_{}_stats.tsv'.format(run_num)
                conf_fpath = os.path.join(group_results_dir, 'instance_' + str(instance_num),
                                          'run_' + str(run_num) + '.ini')

                write_conf(conf_fpath, paths, run_options, misc)
                with open(group_index_fpath, 'ab') as group_index_file:
                    group_index = csv.writer(group_index_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
                    group_index.writerow([instance_num, conf_fpath])

                sim.run(conf_fpath, floader)
                runs_by_instance[instance_num] += 1
