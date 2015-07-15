__author__ = 'Agostino Sturaro'

import os
import netw_creator as nc
import shared_functions as sf

sf.setup_logging('logging_2.json')
conf_path = os.path.abspath('../Simulations/inst_conf.ini')
nc.run(conf_path)
