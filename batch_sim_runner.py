__author__ = 'Agostino Sturaro'

import os
import csv
import logging
import shared_functions as sf
import cascades_sim as sim

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0


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
instances_dir = os.path.normpath('../Simulations/MN_access_pt_diffs_synt')

group_results_dirs = [
    os.path.normpath('../Simulations/MN_access_pt_diffs_synt/rnd_atk/realistic'),

    # os.path.normpath('../Simulations/test_MN/rnd_atk/realistic'),
    # os.path.normpath('../Simulations/test_MN/rnd_atk/sc_th_21'),
    # os.path.normpath('../Simulations/test_MN/rnd_atk/sc_th_201'),
    # os.path.normpath('../Simulations/test_MN/rnd_atk/uniform')
0
    # the part below is used for another kind of plot
    # os.path.normpath('../Simulations/test_MN/subst_atk/realistic')
    # os.path.normpath('../Simulations/test_MN/tran_atk/realistic')
    # os.path.normpath('../Simulations/test_MN/gen_atk/realistic')
]

diff_paths = [
    {
        'netw_inter_fname': 'Inter.graphml'
    # }, {
    #     'netw_inter_fname': 'Inter.graphml'
    # }, {
    #     'netw_inter_fname': 'Inter.graphml'
    # }, {
    #     'netw_inter_fname': 'InterMM.graphml'
    }
]

diff_run_options = [
    {
        'attacked_netw': 'A',
        'attack_tactic': 'random',
        'intra_support_type': 'realistic',
        'inter_support_type': 'realistic',
        'save_death_cause': True
    # }, {
    #     'attacked_netw': 'A',
    #     'attack_tactic': 'random',
    #     'intra_support_type': 'cluster_size',
    #     'min_cluster_size': '21',
    #     'inter_support_type': 'node_interlink'
    # }, {
    #     'attacked_netw': 'A',
    #     'attack_tactic': 'random',
    #     'intra_support_type': 'cluster_size',
    #     'min_cluster_size': '201',
    #     'inter_support_type': 'node_interlink'
    # }, {
    #     'attacked_netw': 'A',
    #     'attack_tactic': 'random',
    #     'intra_support_type': 'giant_component',
    #     'inter_support_type': 'node_interlink'
    }
    # the part below is used for another kind of plot
    # {
    #     'attacked_netw': 'A',
    #     'attack_tactic': 'most_inter_used_distr_subs',
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
]

if len(group_results_dirs) != len(diff_paths) != len(diff_run_options):
    raise ValueError('group_output_dirs, diff_paths and diff_run_options lists should have the same length')

instances_per_type = 3  # used to group instances, must have the same value used for network creation
first_instance_num = 0
last_instance_num = 5

if (1 + last_instance_num - first_instance_num) % instances_per_type:
    raise ValueError('The number of instances is not such that there cannot be the same number of instances for each'
                     'type. Check parameters last_instance_num, first_instance_num, instances_per_type')

seeds = [128, 129]  # used to execute multiple tests on the same network instance

attack_counts = list(range(0, 51, 1))  # values of the independent value of the simulation
# end of user defined variables

for i in range(0, len(diff_paths)):
    paths = diff_paths[i]
    paths['netw_a_fname'] = 'A.graphml'
    paths['netw_b_fname'] = 'B.graphml'
    run_options = diff_run_options[i]
    group_results_dir = group_results_dirs[i]
    sf.ensure_dir_exists(group_results_dir)

    index_fpath = os.path.join(group_results_dir, index_fname)

    # write file used to index statistics files with the final result
    with open(index_fpath, 'ab') as index_file:
        index = csv.writer(index_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        index.writerow(['Instance_type', 'Indep_var_val', 'Results_file'])
        instance_types = (1 + last_instance_num - first_instance_num) / instances_per_type
        for instance_type in range(0, instance_types):
            for point_num, attack_cnt in enumerate(attack_counts):
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
        for point_num, attack_cnt in enumerate(attack_counts):
            run_options['attacks'] = attack_cnt
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

                sim.run(conf_fpath)
                run_num += 1
