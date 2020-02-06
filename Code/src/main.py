import setup_utils #It has to be first line
from setup_utils import MyPool
import os
import shutil
import multiprocessing.pool
import time
import traceback
import logging
import gc

from fpryacc import FPRyacc
from tree_model import TreeModel
from FPTaylor import getFPTaylorResults

def process_file(benchmarks_path, file, output_folder, storage_path, mantissa, exp, range_my_dict, abs_my_dict, goldenModelTime, loadIfExists):
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

        if os.path.exists(output_folder+file_name):
            shutil.rmtree(output_folder+file_name)
        os.makedirs(output_folder+file_name)

        f = open(output_folder + file_name + "/" + file_name + "_CDF_summary.out", "w+")
        f.write("Execution Time:"+str(finalTime)+"s \n\n")

        loadedSamples, values_samples, abs_err_samples, rel_err_samples = T.generate_error_samples(finalTime, file_name, storage_path)
        loadedGolden, values_golden, abs_err_golden, rel_err_golden = T.generate_error_samples(goldenModelTime, file_name, storage_path, loadIfExists)

        T.plot_range_analysis_CDF(loadedGolden, values_samples, values_golden, f, output_folder, file_name, storage_path, range_my_dict.get(file_name))
        T.plot_empirical_error_distribution_CDF(loadedGolden, abs_err_samples, abs_err_golden, f, output_folder, file_name, storage_path, abs_my_dict.get(file_name), rel_my_dict.get(file_name))
        f.flush()
        f.close()

        f = open(output_folder + file_name + "/" + file_name + "_PDF_summary.out", "w+")
        f.write("Execution Time:"+str(finalTime)+"s \n\n")

        T.plot_range_analysis_PDF(loadedGolden, values_samples, values_golden, f, output_folder,file_name, storage_path, range_my_dict.get(file_name))
        T.plot_empirical_error_distribution_PDF(loadedGolden, abs_err_samples, abs_err_golden, f, output_folder, file_name, storage_path, abs_my_dict.get(file_name), rel_my_dict.get(file_name))

        f.flush()
        f.close()

    except Exception as e:
        logging.error(traceback.format_exc())

    finally:
        del values_samples, abs_err_samples, rel_err_samples
        del values_golden, abs_err_golden, rel_err_golden
        gc.collect()

#mantissa with implicit bit of sign
#gmpy2 set precision=p includes also sign bit.
#print(computeLargestPositiveNumber(mantissa, exp))

num_processes=1
setup_utils.init_pacal(8)
mantissa=24
exp=8

home_directory_project=os.getcwd()+"/../"
benchmarks_path=home_directory_project+"benchmark/"
storage_path=home_directory_project+"storage/"
fptaylor_path=home_directory_project+"FPTaylor/"
output_path=home_directory_project+"results/"
fptaylor_exe="/home/roki/GIT/FPTaylor/./fptaylor"
golden_model_time=10
loadIfExists=False
range_my_dict, abs_my_dict, rel_my_dict = getFPTaylorResults(fptaylor_exe, fptaylor_path)

if not len(abs_my_dict) == len(rel_my_dict) and not len(range_my_dict) == len(rel_my_dict):
    print("WARNING!!! Mismatch in FPTaylor")

pool = MyPool(processes=num_processes)

for file in os.listdir(benchmarks_path):
    if file.endswith(".txt"):
        pool.apply_async(process_file, [benchmarks_path, file, output_path, storage_path, mantissa, exp, range_my_dict, abs_my_dict, golden_model_time, loadIfExists])

pool.close()
pool.join()
print("\nAll samples done\n")
