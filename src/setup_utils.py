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
digits_for_discretization=25
digits_for_cdf=4
discretization_points=25
hard_timeout= 5
soft_timeout= hard_timeout * 1000
eps_for_LP= 2**-20
divisions_SMT_pruning_operation=3
valid_for_exit_SMT_pruning_operation=2
divisions_SMT_pruning_error=10
valid_for_exit_SMT_pruning_error=9
gap_cdf_regularizer = 1.0/discretization_points
golden_model_time=10

recursion_limit_for_pruning_operation=5
recursion_limit_for_pruning_error=20

delta_error_computation=10*(2**-24)
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
