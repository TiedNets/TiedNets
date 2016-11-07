__author__ = 'Agostino Sturaro'

import os
import netw_creator as nc
import shared_functions as sf

# This is a simple file to run custom tests

this_dir = os.path.normpath(os.path.dirname(__file__))
os.chdir(this_dir)

logging_conf_fpath = os.path.normpath('logging_base_conf.json')
netw_conf_fpath = os.path.normpath('../Simulations/temp/config_1.ini')

sf.setup_logging(logging_conf_fpath)
conf_path = os.path.abspath(netw_conf_fpath)
nc.run(conf_path)
