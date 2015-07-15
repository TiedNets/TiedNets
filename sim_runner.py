__author__ = 'Agostino Sturaro'

import os
import filecmp
import shutil
import shared_functions as sf
import cascades_sim as cs

# given
this_dir = os.path.normpath(os.path.dirname(__file__))
logging_conf_fpath = os.path.join(this_dir, 'logging_base_conf.json')
sim_conf_fpath = 'test_sets/ex_1_full/run_realistic.ini'
exp_log_fpath = 'test_sets/ex_1_full/exp_log_realistic.txt'

# when
os.chdir(this_dir)
sf.setup_logging(logging_conf_fpath)
cs.run(sim_conf_fpath)

# then
assert filecmp.cmp('log.txt', exp_log_fpath, False)  # assuming UNIX EOLs are used

# tear down
# shutil.rmtree(os.path.join(this_dir, os.path.normpath('test_sets/ex_1_full/res_realistic')))