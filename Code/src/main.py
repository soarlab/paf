import setup_utils #It has to be first line, do not remove

from setup_utils import home_directory_project, benchmarks_path,\
    storage_path,fptaylor_path,output_path,fptaylor_exe,pran_exe,\
    golden_model_time, loadIfExists, storeIfDoesnExist, MyPool

import warnings
import os
import shutil
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

warnings.warn("Mantissa with implicit bit of sign. In gmpy2 set precision=p includes also sign bit.\n")

mantissa=24
exp=8

range_my_dict, abs_my_dict, rel_my_dict = getFPTaylorResults(fptaylor_exe, fptaylor_path)

pool = MyPool(processes=setup_utils.num_processes)

for file in os.listdir(benchmarks_path):
    if file.endswith(".txt"):
        pool.apply_async(process_file, [benchmarks_path, file, output_path, storage_path, mantissa, exp, range_my_dict, abs_my_dict, golden_model_time, loadIfExists])

pool.close()
pool.join()
print("\nAll samples done\n")
