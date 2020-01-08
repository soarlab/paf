import matplotlib.pyplot
from fpryacc import *
from tree_model import TreeModel
import time
import os
import shutil
from distributions import *

matplotlib.pyplot.close("all")
mantissa=24
exp=8

testExp()
#mantissa with implicit bit of sign
#gmpy2 set precision=p includes also sign bit.
#print(computeLargestPositiveNumber(mantissa, exp))
benchmarks_path="./benchmarks/"

for file in os.listdir(benchmarks_path):
    if file.endswith(".txt"):
        print(file)
        f = open(benchmarks_path+file,"r")
        file_name = file.split(".")[0]
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
        T.collectInfoAboutDistribution(f)
        T.plot_range_analysis(f, finalTime,benchmarks_path,file_name)
        #T.plot_empirical_error_distribution(100000,"error_dist")
        f.close()

print("\nDone\n")
