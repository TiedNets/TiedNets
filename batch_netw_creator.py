import os
import json
import logging
import netw_creator as nc
import shared_functions as sf
from collections import OrderedDict

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0

__author__ = 'Agostino Sturaro'

# global variables
logger = logging.getLogger(__name__)


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


def run(batch_conf_fpath):
    with open(batch_conf_fpath) as batch_conf_file:
        batch_conf = json.load(batch_conf_file, object_pairs_hook=OrderedDict)

    base_dir = os.path.normpath(batch_conf['base_dir'])

    seeds = sf.read_variable_values_in_conf(batch_conf, 'seeds')
    logger.info('seeds = {}'.format(seeds))

    roles_fpaths_a = None
    if 'preassigned_roles_fpath' in batch_conf['build_a']:
        roles_fpaths_a = sf.read_variable_paths_in_conf(batch_conf['build_a'], 'preassigned_roles_fpath')

        if len(seeds) != len(roles_fpaths_a):
            raise ValueError('The number of seeds must be the same as the number of preassigned roles file paths. '
                             'Check the configuration options "seeds" and "preassigned_roles_fpath".')

    # clean the destination folder if specified, otherwise ask if necessary
    clean_output_dir = False
    if 'clean_output_dir' in batch_conf['misc']:
        clean_output_dir = batch_conf['misc']['clean_output_dir']
    if clean_output_dir is True:
        sf.ensure_dir_clean(base_dir, True, False)
    else:
        sf.ensure_dir_clean(base_dir, True, True)

    a_opts = batch_conf['build_a']
    b_opts = batch_conf['build_b']
    inter_opts = batch_conf['build_inter']
    misc_opts = batch_conf['misc']

    for instance_num in range(0, len(seeds)):
        seed = seeds[instance_num]
        logger.debug('Current seed: {}'.format(seed))
        misc_opts['seed'] = seed

        if roles_fpaths_a is not None:
            a_opts['preassigned_roles_fpath'] = roles_fpaths_a[instance_num]

        instance_dir = os.path.join(base_dir, 'instance_{}'.format(instance_num))
        conf_fpath = os.path.join(base_dir, 'config_{}.ini'.format(instance_num))
        write_conf(instance_dir, conf_fpath, a_opts, b_opts, inter_opts, misc_opts)
        nc.run(conf_fpath)
