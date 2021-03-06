import os
import csv
import logging
import file_loader as fl
import shared_functions as sf
import cascades_sim as sim

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0

__author__ = 'Agostino Sturaro'


def write_conf(conf_fpath, paths, run_options):
    config = ConfigParser()

    config.add_section('paths')
    for opt_name in paths:
        config.set('paths', opt_name, paths[opt_name])

    config.add_section('run_opts')
    for opt_name in run_options:
        config.set('run_opts', opt_name, run_options[opt_name])

    sf.ensure_dir_exists(os.path.dirname(os.path.realpath(conf_fpath)))

    with open(conf_fpath, 'w') as conf_file:
        config.write(conf_file)


this_dir = os.path.normpath(os.path.dirname(__file__))
os.chdir(this_dir)
sf.setup_logging('logging_base_conf.json')
logger = logging.getLogger(__name__)

# begin of user defined variables
index_fname = '_index.tsv'
# instances_dir = os.path.normpath('../Simulations/MN_nets/1cc_1ap')
# instances_dir = os.path.normpath('../Simulations/synthetic_nets/1cc_1ap')
instances_dir = os.path.normpath('../Simulations/centrality/1cc_1ap')

group_results_dirs = [
    # os.path.normpath('../Simulations/MN_nets/1cc_1ap/rnd_atk/realistic'),
    # os.path.normpath('../Simulations/MN_nets/1cc_1ap/rnd_atk/sc_th_21'),
    # os.path.normpath('../Simulations/MN_nets/1cc_1ap/rnd_atk/sc_th_210'),
    # os.path.normpath('../Simulations/MN_nets/1cc_1ap/rnd_atk/uniform')

    # os.path.normpath('../Simulations/synthetic_nets/1cc_1ap/rnd_atk/realistic'),
    # os.path.normpath('../Simulations/synthetic_nets/1cc_1ap/rnd_atk/sc_th_21'),
    # os.path.normpath('../Simulations/synthetic_nets/1cc_1ap/rnd_atk/sc_th_210'),
    # os.path.normpath('../Simulations/synthetic_nets/1cc_1ap/rnd_atk/uniform')

    # the part below is used for another kind of plot
    # os.path.normpath('../Simulations/MN_nets/1cc_1ap/deg_atks/inter_subst_atk/realistic'),
    # os.path.normpath('../Simulations/MN_nets/1cc_1ap/deg_atks/intra_subst_atk/realistic'),
    # os.path.normpath('../Simulations/MN_nets/1cc_1ap/deg_atks/intra_tran_atk/realistic')

    # os.path.normpath('../Simulations/MN_nets/1cc_1ap/deg_atks/intra_gen_atk/realistic')

    os.path.normpath('../Simulations/centrality/1cc_1ap/1_at_once/realistic_inter')
]

diff_paths = [
    {
        #     'netw_inter_fname': 'Inter.graphml'
        # }, {
        #     'netw_inter_fname': 'Inter.graphml',
        # }, {
        'netw_a_fname': 'A.graphml',
        'netw_b_fname': 'B.graphml',
        'netw_inter_fname': 'Inter.graphml',
        'netw_union_fname': 'UnionAB.graphml',
        'ml_stats_fpath': '../Simulations/centrality/1cc_1ap/1_at_once/realistic/ml_stats_0_inter.tsv'
        # }, {
        #     'netw_inter_fname': 'InterMM.graphml'
    }
]

diff_run_options = [
    # {
    #     'attacked_netw': 'A',
    #     'attack_tactic': 'random',
    #     'intra_support_type': 'realistic',
    #     'inter_support_type': 'realistic',
    #     'save_death_cause': True
    # }, {
    #     'attacked_netw': 'A',
    #     'attack_tactic': 'random',
    #     'intra_support_type': 'cluster_size',
    #     'min_cluster_size': 21,
    #     'inter_support_type': 'node_interlink',
    #     'save_death_cause': True
    # }, {
    #     'attacked_netw': 'A',
    #     'attack_tactic': 'random',
    #     'intra_support_type': 'cluster_size',
    #     'min_cluster_size': 210,
    #     'inter_support_type': 'node_interlink',
    #     'save_death_cause': True
    # }, {
    #     'attacked_netw': 'A',
    #     'attack_tactic': 'random',
    #     'intra_support_type': 'giant_component',
    #     'inter_support_type': 'node_interlink',
    #     'save_death_cause': True
    # }
    # the part below is used for another kind of plot
    # {
    #     'attacked_netw': 'A',
    #     'attack_tactic': 'most_inter_used_distr_subs',
    #     'intra_support_type': 'realistic',
    #     'inter_support_type': 'realistic',
    #     'save_death_cause': True
    # },{
    #     'attacked_netw': 'A',
    #     'attack_tactic': 'most_intra_used_distr_subs',
    #     'intra_support_type': 'realistic',
    #     'inter_support_type': 'realistic',
    #     'save_death_cause': True
    # }, {
    #     'attacked_netw': 'A',
    #     'attack_tactic': 'most_intra_used_transm_subs',
    #     'intra_support_type': 'realistic',
    #     'inter_support_type': 'realistic',
    #     'save_death_cause': True
    # }, {
    #     'attacked_netw': 'A',
    #     'attack_tactic': 'most_intra_used_generators',
    #     'intra_support_type': 'realistic',
    #     'inter_support_type': 'realistic',
    #     'save_death_cause': True
    # }
    {
        'attacked_netw': 'Both',
        'attack_tactic': 'indegree_centrality_rank',
        'calc_centrality_on': 'netw_inter',
        'intra_support_type': 'realistic',
        'inter_support_type': 'realistic',
        'save_death_cause': True,
        'attacks': 1
    }
]

if len(group_results_dirs) != len(diff_paths) != len(diff_run_options):
    raise ValueError('group_output_dirs, diff_paths and diff_run_options lists should have the same length')

# TODO: fix description
# this is the older batch runner that combined 4 sets of parameters instead of 3, and organized files differently

# simulations can be done on groups of instances, and each
# here's how to use these parameters, if there is a single group of 3 similar instances
# first_instance_num = 0, instances_per_type = 2, last_instance_num = 3
instances_per_type = 1  # used to group instances, must have the same value used for network creation
first_instance_num = 0  # usually 0, unless you want to skip a group, then it should be divisible by instances_per_type
last_instance_num = 0  # inclusive,

if (1 + last_instance_num - first_instance_num) % instances_per_type:
    raise ValueError('Incoherent values for parameters  first_instance_num, last_instance_num and instances_per_type'
                     'Make it so that ')

# seeds = list(range(128, 138, 1))  # used to execute multiple tests on the same network instance
seeds = [1]

# attack_counts = list(range(0, 151, 5))  # values of the independent value of the simulation
# attack_counts = list(range(0, 61, 5)) + [69]  # values of the independent value of the simulation
# indep_var_name = 'attacks'  # name of the independent variable of the simulation
indep_var_name = 'min_rank'  # name of the independent variable of the simulation
indep_var_vals = list(range(0, 2000, 1))  # values of the independent variable of the simulation
# end of user defined variables

floader = fl.FileLoader()

for i in range(0, len(diff_paths)):
    paths = diff_paths[i]
    run_options = diff_run_options[i]
    group_results_dir = group_results_dirs[i]
    sf.ensure_dir_exists(group_results_dir)

    index_fpath = os.path.join(group_results_dir, index_fname)

    # write file used to index statistics files with the final result
    with open(index_fpath, 'wb') as index_file:
        index = csv.writer(index_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        index.writerow(['Instance_type', 'Indep_var_val', 'Results_file'])
        instance_types = (1 + last_instance_num - first_instance_num) / instances_per_type
        for instance_type in range(0, instance_types):
            for point_num, attack_cnt in enumerate(indep_var_vals):
                end_stats_fpath = os.path.join(group_results_dir,
                                               'group_{}_stats_{}.tsv'.format(instance_type, point_num))
                index.writerow([instance_type, attack_cnt, end_stats_fpath])

    # outer cycle ranging over different network instances
    instance_num = 0
    for instance_num in range(first_instance_num, last_instance_num + 1, 1):
        run_num = 0
        instance_type = int(float(instance_num) / instances_per_type)  # integer division, compatible with v3
        paths['netw_dir'] = os.path.join(instances_dir, 'instance_{}'.format(instance_num))

        # inner cycle ranging over values of the independent variable
        for point_num, attack_cnt in enumerate(indep_var_vals):
            run_options[indep_var_name] = attack_cnt
            paths['end_stats_fpath'] = os.path.join(group_results_dir,
                                                    'group_{}_stats_{}.tsv'.format(instance_type, point_num))

            for seed in seeds:
                run_options['seed'] = seed
                paths['run_stats_fname'] = 'run_{}_stats.tsv'.format(run_num)
                paths['results_dir'] = os.path.join(group_results_dir, 'instance_' + str(instance_num),
                                                    'run_' + str(run_num))
                conf_fpath = os.path.join(group_results_dir, 'instance_' + str(instance_num),
                                          'run_' + str(run_num) + '.ini')
                write_conf(conf_fpath, paths, run_options)

                # TODO: build system using this to aggregate stats
                group_index_fpath = os.path.join(group_results_dir,
                                                 'group_{}_index_{}.tsv'.format(instance_type, point_num))
                with open(group_index_fpath, 'ab') as group_index_file:
                    group_index = csv.writer(group_index_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
                    group_index.writerow([os.path.join(paths['results_dir'], paths['run_stats_fname'])])

                sim.run(conf_fpath, floader)
                run_num += 1
