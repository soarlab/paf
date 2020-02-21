import os
# disable openblas threading
# This must be done before importing numpy
par=1
os.environ["OPENBLAS_NUM_THREADS"] = str(par)

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
golden_model_time=7200
global_interpolate=True
loadIfExists=True
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
