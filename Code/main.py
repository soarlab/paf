import matplotlib.pyplot
from fpryacc import *
from tree_model import TreeModel
import time
import os
import shutil
from distributions import *
from FPTaylor import *

matplotlib.pyplot.close("all")
mantissa=24
exp=8
#mantissa with implicit bit of sign
#gmpy2 set precision=p includes also sign bit.
#print(computeLargestPositiveNumber(mantissa, exp))
benchmarks_path="./benchmarks/"
executeOnBenchmarks("/home/roki/GIT/FPTaylor/./fptaylor", "/home/roki/GIT/FPTaylor/benchmarks/probability/")
abs_my_dict=getAbsoluteError("/home/roki/GIT/FPTaylor/benchmarks/probability/results")
rel_my_dict=getRelativeError("/home/roki/GIT/FPTaylor/benchmarks/probability/results")
range_my_dict=getBounds("/home/roki/GIT/FPTaylor/benchmarks/probability/results")
if not len(abs_my_dict) == len(rel_my_dict) and not len(range_my_dict) == len(rel_my_dict):
    print("WARNING!!! Mismatch ")
for file in os.listdir(benchmarks_path):
    if file.endswith(".txt"):
        try:
            print(file)
            f = open(benchmarks_path+file,"r")
            file_name = (file.split(".")[0]).lower()
            text = f.read()
            text = text[:-1]
            f.close()
            myYacc=FPRyacc(text, False)
            start_time = time.time()
            T = TreeModel(myYacc, mantissa, exp, 50)
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
            T.plot_empirical_error_distribution(f, finalTime,benchmarks_path,file_name, abs_my_dict.get(file_name), rel_my_dict.get(file_name))
            f.close()
        except Exception as e:
            print(e)

print("\nDone\n")