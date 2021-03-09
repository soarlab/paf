import setup_utils  # It has to be first line, do not remove

from setup_utils import home_directory_project, benchmarks_path, \
    storage_path, fptaylor_path, output_path, fptaylor_exe, pran_exe, \
    golden_model_time, loadIfExists, storeIfDoesnExist, MyPool

from plotting import plot_range_analysis_CDF, plot_range_analysis_PDF, \
    plot_error_analysis_PDF, plot_abs_error_analysis_CDF

import warnings
import os
import shutil
import time
import traceback
import logging
import gc
import ntpath
import matplotlib

from fpryacc import FPRyacc
from plotting import plot_error_analysis_CDF
from tree_model import TreeModel
from FPTaylor import getFPTaylorResults
from tests.tests import test_Approx_Operations

#test_Approx_Operations()

def process_file(file, mantissa, exp, range_my_dict, abs_my_dict):
    try:
        print("\n\n\n\n"+file+"\n\n\n\n\n")
        f = open(file, "r")
        file_name = (ntpath.basename(file).split(".")[0]).lower()  # (file.split(".")[0]).lower()
        text = f.read()
        text = text[:-1]
        f.close()
        myYacc = FPRyacc(text, False)
        start_time = time.time()
        T = TreeModel(myYacc, mantissa, exp, [40, 10], 50, 250000)#, error_model="typical", dependent_mode="p-box")
        end_time = time.time()
        print("Exe time --- %s seconds ---" % (end_time - start_time))
        finalTime = end_time - start_time

        if os.path.exists(output_path + file_name):
            shutil.rmtree(output_path + file_name)
        os.makedirs(output_path + file_name)

        #loadedSamples, values_samples, abs_err_samples, rel_err_samples = T.generate_error_samples(5, file_name)
        loadedGolden, values_golden, abs_err_golden, rel_err_golden, err_golden = T.generate_error_samples(golden_model_time,
                                                                                               file_name, golden=True)

        with open(output_path + file_name + "/" + file_name + "_constraints_log.out", "w+") as log:
            print(T.logging_constraints, file=log)

        f = open(output_path + file_name + "/" + file_name + "_CDF_summary.out", "w+")
        f.write("Execution Time:" + str(finalTime) + "s \n\n")
        plot_range_analysis_CDF(T, loadedGolden, values_golden, f, file_name, range_my_dict.get(file_name))
        plot_abs_error_analysis_CDF(T, loadedGolden, abs_err_golden, f, file_name, abs_my_dict.get(file_name), rel_my_dict.get(file_name))
        #plot_error_analysis_CDF(T, loadedGolden, err_golden, f, file_name, abs_my_dict.get(file_name), rel_my_dict.get(file_name))
        #plot_abs_error_analysis_CDF(T.relative_err_distr, loadedGolden, rel_err_samples, rel_err_golden, f, file_name, abs_my_dict.get(file_name), rel_my_dict.get(file_name))
        f.flush()
        f.close()

        f = open(output_path + file_name + "/" + file_name + "_PDF_summary.out", "w+")
        f.write("Execution Time:" + str(finalTime) + "s \n\n")
        #plot_range_analysis_PDF(T.final_quantized_distr, loadedGolden, values_golden, f, file_name, range_my_dict.get(file_name))
        #plot_error_analysis_PDF(T.abs_err_distr, loadedGolden, abs_err_samples, abs_err_golden, f, file_name, abs_my_dict.get(file_name), rel_my_dict.get(file_name))
        f.flush()
        f.close()


    except Exception as e:
        logging.error(traceback.format_exc())

    finally:
        del values_golden, abs_err_golden, rel_err_golden
        gc.collect()
        matplotlib.pyplot.close("all")

warnings.warn("Mantissa with implicit bit of sign. In gmpy2 set precision=p includes also sign bit. (e.g. Float32 is mantissa=24 and exp=8)\n")

mantissa = 53
exp = 11
range_my_dict, abs_my_dict, rel_my_dict = getFPTaylorResults(fptaylor_exe, fptaylor_path)

for file in os.listdir(benchmarks_path):
    try:
        if file.endswith(".txt"):
            process_file(benchmarks_path+file, mantissa, exp, range_my_dict, abs_my_dict)
    except:
        print("CRASH: "+str(file))
print("\nDone with sample\n")
