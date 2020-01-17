import os
# disable openblas threading
# This must be done before importing numpy

par=1
os.environ["OPENBLAS_NUM_THREADS"] = str(par)

import utils
utils.init_pacal()


import matplotlib.pyplot
from fpryacc import *
from tree_model import TreeModel
import time
import sys
import multiprocessing
import multiprocessing.pool
import time
from FPTaylor import *
import traceback
import logging
import utils

class NoDaemonProcess(multiprocessing.Process):
     # make 'daemon' attribute always return False
     def _get_daemon(self):
         return False
     def _set_daemon(self, value):
        pass
     daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def process_file(benchmarks_path, file, mantissa, exp, range_my_dict, abs_my_dict):
    try:
        print(file)
        f = open(benchmarks_path+file,"r")
        file_name = (file.split(".")[0]).lower()
        text = f.read()
        text = text[:-1]
        f.close()
        myYacc=FPRyacc(text, False)
        start_time = time.time()
        T = TreeModel(myYacc, mantissa, exp, 100, 250000)
        end_time = time.time()
        print("Exe time --- %s seconds ---" % (end_time - start_time))
        finalTime=end_time-start_time
        if os.path.exists(benchmarks_path+file_name):
            shutil.rmtree(benchmarks_path+file_name)
        os.makedirs(benchmarks_path+file_name)
        f = open(benchmarks_path + file_name + "/" + file_name + "_summary.out", "w+")
        f.write("Execution Time:"+str(finalTime)+"s \n\n")
        T.collectInfoAboutDistribution(f, T.tree.root_value[2], "Range Analysis on Round(distr)")
        T.plot_range_analysis(f, finalTime,benchmarks_path,file_name, range_my_dict.get(file_name))
        #T.plot_empirical_error_distribution(f, finalTime,benchmarks_path,file_name, abs_my_dict.get(file_name), rel_my_dict.get(file_name))
        f.close()
    except Exception as e:
        logging.error(traceback.format_exc())


matplotlib.pyplot.close("all")
mantissa=24
exp=8

#mantissa with implicit bit of sign
#gmpy2 set precision=p includes also sign bit.
#print(computeLargestPositiveNumber(mantissa, exp))

benchmarks_path="./benchmarks/"
executeOnBenchmarks("/home/roki/GIT/FPTaylor/./fptaylor", "./FPTaylor/")
abs_my_dict=getAbsoluteError("./FPTaylor/results")
rel_my_dict=getRelativeError("./FPTaylor/results")
range_my_dict=getBounds("./FPTaylor/results")
if not len(abs_my_dict) == len(rel_my_dict) and not len(range_my_dict) == len(rel_my_dict):
    print("WARNING!!! Mismatch ")

pool = MyPool(processes=1)#int(os.cpu_count() / par))

for file in os.listdir(benchmarks_path):
    if file.endswith(".txt"):
        pool.apply_async(process_file, [benchmarks_path, file, mantissa, exp, range_my_dict, abs_my_dict])

pool.close()
pool.join()
print("all samples done")