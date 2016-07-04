import os
import sys
import json
import subprocess

# each of these processes must have its own configuration file
# if we need them to run concurrently, all of their output files must be different
concurrent_procs = 6

# check that all configuration files exist and are valid json
# for batch_no in range(concurrent_procs):
for batch_no in range(20, 26):
    batch_conf_fpath = os.path.normpath('../Simulations/test_mp/group_{}.json'.format(batch_no))
    if os.path.exists(batch_conf_fpath):
        try:
            with open(batch_conf_fpath, 'rt') as f:
                config = json.load(f)
        except Exception as exc:
            raise ValueError('The batch configuration file {} is not a valid json file'.format(batch_conf_fpath, exc))
    else:
        raise RuntimeError('The batch configuration file {} does not exist'.format(batch_conf_fpath))

procs = []
# spawn the processes
# for batch_no in range(concurrent_procs):
for batch_no in range(20, 26):
    batch_conf_fpath = os.path.normpath('../Simulations/test_mp/group_{}.json'.format(batch_no))
    proc = subprocess.Popen([sys.executable, 'batch_sim_runner_2.py', str(batch_no), batch_conf_fpath])
    procs.append(proc)

# wait for child processes to finish
for proc in procs:
    proc.wait()
