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

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
