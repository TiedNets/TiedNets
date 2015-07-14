__author__ = 'sturaroa'

import os
import filecmp
import shared_functions as sf
import cascades_sim as cs

logging_conf_fpath = os.path.normpath('C:/Users/sturaroa/Documents/TiedNets/logging_base_conf.json')
sim_conf_fpath = os.path.normpath('C:/Users/sturaroa/Documents/Simulations/exp_1000n_test_2/rnd_atk/realistic/instance_0/run_0.ini')
# exp_log_fpath = os.path.normpath('C:/Users/sturaroa/Documents/Simulations/test_0/ex_1_full/exp_log_realistic.txt')

sf.setup_logging(logging_conf_fpath)
cs.run(sim_conf_fpath)

# print(filecmp.cmp('log.txt', exp_log_fpath, False))
