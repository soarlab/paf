import os
# disable openblas threading
# This must be done before importing numpy
import psutil
par=1
os.environ["OPENBLAS_NUM_THREADS"] = str(par)
os.environ["PATH"] = "/usr/bin" + os.pathsep + os.environ["PATH"]


import sys
import argparse
class MyParser(argparse.ArgumentParser):
   def error(self, message):
      sys.stderr.write('ERROR: %s\n' % message)
      self.print_help()
      sys.exit(1)



parser = MyParser(description='PAF - Probalistic Analysis of Floating-point arithmetic',
                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('res', type=str, metavar='<path_to_program>',
                    help='the path of the file you want to verify. '
                         'In case the path leads to a folder, PAF is going to process all the files one by one.')
parser.add_argument('-m', type=int, metavar='<mantissa_format>',
                    help='Mantissa format in bits', default=53)
parser.add_argument('-e', type=int, metavar='<exponent_format>',
                    help='Exponent format in bits', default=11)
parser.add_argument('-d', type=int, metavar='<discretization_size>',
                    help='Size of the DS structure', default=50)
parser.add_argument('-tgc', type=int, metavar='<timeout_gelpia_cnstrs>',
                    help='Timeout for Gelpia when using constraints', default=180)
parser.add_argument('-z', action='store_true', help='Use exclusive z3 (no dreal)')
parser.add_argument('-prob', type=str, metavar='<confidence_interval>',
                    help='Confidence interval (e.g. 0.95, 0.99, 0.999999)', default="1")

args = parser.parse_args()

from decimal import getcontext
getcontext().prec = 500

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
benchmarks_path=args.res #home_directory_project+"benchmarks_tmp/"
mantissa_format=args.m
exponent_format=args.e

storage_path=home_directory_project+"storage/"
fptaylor_path=home_directory_project+"FPTaylor/"
output_path=home_directory_project+"results/"
fptaylor_exe="./fptaylor"
pran_exe=""
num_processes_dependent_operation=int(multiprocessing.cpu_count())
#memory_limit_optimization=((psutil.virtual_memory().total//1024)//1024)

mpfr_proxy_precision=2000
use_powers_of_two_spacing=False
custom_spacing=True
sigma_for_normal_distribution=1  #0.01
sigma_for_exponential_distribution=0.01
scale_for_rayleigh_distribution=2


digits_for_input_discretization=20
digits_for_range=50
digits_for_input_cdf=15
digits_for_Z3_cdf=20

discretization_points= args.d
hard_timeout= 10
soft_timeout= hard_timeout * 1000
eps_for_LP= 2**-20
divisions_SMT_pruning_operation=10
valid_for_exit_SMT_pruning_operation=6
divisions_SMT_pruning_error=10
valid_for_exit_SMT_pruning_error=9
gap_cdf_regularizer = 1.0/discretization_points
golden_model_time=60
timeout_gelpia_constraints=args.tgc
timeout_gelpia_standard=60
timeout_optimization_problem=300
round_constants_to_nearest=True
constraints_probabilities=args.prob #"0.999999"

abs_prefix="ABS_"
SMT_exponent_function_name= "find_exponent"
GELPHIA_exponent_function_name= "floor_power2"
path_to_gelpia_executor="python3.7 ./tmp/gelpia/bin/gelpia.py "
path_to_gelpia_constraints_executor="python3.7 ./tmp/gelpia_constraints/bin/gelpia.py "
use_z3_when_constraints_gelpia=args.z

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
