import os
# disable openblas threading
# This must be done before importing numpy
par=1
os.environ["OPENBLAS_NUM_THREADS"] = str(par)

from decimal import getcontext
getcontext().prec = 100

import multiprocessing
from multiprocessing import pool
import pacal

def init_pacal(num_threads):
    '''Limit pacal threads'''
    pacal.params.general.parallel = True
    pacal.params.general.nprocs = num_threads

num_threads=8
init_pacal(num_threads)
num_processes=int(multiprocessing.cpu_count()/num_threads)

home_directory_project=os.getcwd()+"/"
benchmarks_path=home_directory_project+"benchmarks/"
storage_path=home_directory_project+"storage/"
fptaylor_path=home_directory_project+"FPTaylor/"
output_path=home_directory_project+"results/"
fptaylor_exe="/home/roki/GIT/FPTaylor/./fptaylor"
pran_exe=""
num_processes_dependent_operation=int(multiprocessing.cpu_count())

mpfr_proxy_precision=200

digits_for_input_discretization=20
digits_for_range=25
digits_for_cdf=10
digits_for_Z3_cdf=6

discretization_points=5
hard_timeout= 10
soft_timeout= hard_timeout * 1000
eps_for_LP= 2**-20
divisions_SMT_pruning_operation=3
valid_for_exit_SMT_pruning_operation=2
divisions_SMT_pruning_error=10
valid_for_exit_SMT_pruning_error=9
gap_cdf_regularizer = 1.0/discretization_points
golden_model_time=10
timeout_gelpia=10
round_constants_to_nearest=True

abs_prefix="ABS_"
SMT_exponent_function_name= "find_exponent"
GELPHIA_exponent_function_name= "floor_power2"
path_to_gelpia_executor="python3.7 /home/roki/GIT/FPTaylor/gelpia/bin/gelpia.py "
path_to_gelpia_constraints_executor="python3.7 /home/roki/GIT/gelpia_constraints_orig/bin/gelpia.py "

recursion_limit_for_pruning_operation=10
recursion_limit_for_pruning_error=20

global_interpolate=True
loadIfExists=False
storeIfDoesnExist=False


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
