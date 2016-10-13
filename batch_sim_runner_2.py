import os
import sys
import csv
import json
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


def pick_conf_values(config, opt_name):
    values = None
    if config[opt_name]['pick'] == 'range':
        start = config[opt_name]['start']
        stop = config[opt_name]['stop']
        if 'step' in config[opt_name]:
            step = config[opt_name]['step']
        else:
            step = 1
        values = range(start, stop, step)
    elif config[opt_name]['pick'] == 'specified':
        if 'single_value' in config[opt_name]:
            values = [config[opt_name]['single_value']]
        elif 'list_of_values' in config[opt_name]:
            values = config[opt_name]['list_of_values']
    return values


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

# A "batch" is a set of simulations to be executed. Unlike simulation groups, it does not have to contain simulations
# using similar parameters. A batch can contain:
# - simulations of the same "group", all of that sim_group, or just some of them
# - simulations from different groups

this_dir = os.path.normpath(os.path.dirname(__file__))
os.chdir(this_dir)

batch_no = int(sys.argv[1])
batch_conf_fpath = sys.argv[2]

# begin of user defined variables

with open(batch_conf_fpath) as batch_conf_file:
    batch_conf = json.load(batch_conf_file)

logging.config.dictConfig(batch_conf['logging_config'])
logger = logging.getLogger(__name__)

instances_dir = os.path.normpath(batch_conf['instances_dir'])  # parent directory of the instances directories
base_configs = batch_conf['base_configs']

# TODO: refine this to a list of instances
# recall the value of the parameter sim_group_size used in creating batches of networks
first_instance = batch_conf['first_instance']  # usually 0, unless you want to skip a group
last_instance = batch_conf['last_instance']  # exclusive, should be divisible by sim_group_size
logger.info('first_instance = {}'.format(first_instance))
logger.info('last_instance = {}'.format(last_instance))

# This script is used to run multiple simulations on a set of instances. At its core is a loop that generates a
# configuration file with a different combination of parameters and executes a simulation based on it.
# If we want to use a group of simulations to draw a 2D plot, showing the behavior obtained by changing a variable, then
# we need to specify the name of the independent variable and the values we want it to assume.

# name of the independent variable of the simulation
indep_var_name = batch_conf['indep_var_name']
# indep_var_vals = list(range(0, 3, 1))
# indep_var_vals = sorted(random.sample(range(1, 200), 50))
#indep_var_vals = range(1, 26)
#indep_var_vals = range(26, 76)
# indep_var_vals = range(1, 76)
logger.info('indep_var_name = {}'.format(indep_var_name))

# values of the independent value of the simulation
indep_var_vals = pick_conf_values(batch_conf, 'indep_var_vals')
# indep_var_vals = list(range(0, 3, 1))
# indep_var_vals = sorted(random.sample(range(1, 200), 50))
#indep_var_vals = range(1, 26)
#indep_var_vals = range(26, 76)
# indep_var_vals = range(1, 76)
#indep_var_vals = range(1, 51)
logger.info('indep_var_vals = {}'.format(indep_var_vals))
# indep_var_vals = list(range(0, 61, 5)) + [69]  # values of the independent value of the simulation
# indep_var_name = 'min_rank'  # name of the independent variable of the simulation, in the run_opts section
# indep_var_vals = list(range(0, 2000, 1))  # values of the independent variable of the simulation

# seeds used to execute multiple tests on the same network instance
seeds = pick_conf_values(batch_conf, 'seeds')
#seeds = list(range(50, 200, 1))
#seeds = list(range(0, 50, 1))
logger.info('seeds = {}'.format(seeds))
# end of user defined variables

floader = fl.FileLoader()

for sim_group in range(0, len(base_configs)):
    paths = base_configs[sim_group]['paths']
    run_options = base_configs[sim_group]['run_opts']
    group_results_dir = os.path.normpath(paths['results_dir'])
    sf.ensure_dir_exists(group_results_dir)
    misc = base_configs[sim_group]['misc']
    misc['sim_group'] = sim_group
    paths['batch_conf_fpath'] = batch_conf_fpath
    run_num_by_inst = {}  # number of simulations we ran for each instance i

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
            # if it's the first simulation we run for this instance, then take note
            if instance_num not in run_num_by_inst:
                run_num_by_inst[instance_num] = 0

            # inner cycle ranging over different seeds
            for seed in seeds:
                logger.debug("Batch {}) {} value, seed {}".format(batch_no, var_value, seed))
                # pick up the simulation number (n-th time we run a simulation on this instance)
                run_num = run_num_by_inst[instance_num]
                misc['instance'] = instance_num  # mark in the configuration file the number of this instance
                misc['run'] = run_num
                paths['netw_dir'] = os.path.join(instances_dir, 'instance_{}'.format(instance_num))  # input
                run_options['seed'] = seed
                paths['results_dir'] = os.path.join(group_results_dir, 'instance_' + str(instance_num),
                                                    'run_' + str(run_num))
                paths['run_stats_fname'] = 'run_{}_stats.tsv'.format(run_num)
                conf_fpath = os.path.join(group_results_dir, 'instance_' + str(instance_num),
                                          'run_' + str(run_num) + '.ini')

                write_conf(conf_fpath, paths, run_options, misc)
                with open(group_index_fpath, 'ab') as group_index_file:
                    group_index = csv.writer(group_index_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
                    group_index.writerow([instance_num, conf_fpath])

                sim.run(conf_fpath, floader)  # run the simulation
                run_num_by_inst[instance_num] += 1  # next simulation for this instance will be number + 1
