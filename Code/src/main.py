import setup_utils #It has to be first line, do not remove

from setup_utils import home_directory_project, benchmarks_path,\
    storage_path,fptaylor_path,output_path,fptaylor_exe,pran_exe,\
    golden_model_time, loadIfExists, storeIfDoesnExist, MyPool

from plotting import plot_range_analysis_CDF, plot_range_analysis_PDF, \
    plot_error_analysis_PDF, plot_error_analysis_CDF

import warnings
import os
import shutil
import time
import traceback
import logging
import gc
import ntpath

from fpryacc import FPRyacc
from tree_model import TreeModel, copy_tree
from FPTaylor import getFPTaylorResults

def process_file(xs, file, mantissa, exp, range_my_dict, abs_my_dict):
    try:
        print(file)
        f = open(file,"r")
        #file_name = (ntpath.basename(file).split(".")[0]).lower()
        text = f.read()
        text = text[:-1]
        f.close()
        myYacc=FPRyacc(text, False)

        start_time = time.time()

        T = TreeModel(xs, myYacc, mantissa, exp, 100, 250000)

        end_time = time.time()

        file_name = (ntpath.basename(file).split(".")[0]).lower()+str(T.counter+1)

        print("Exe time --- %s seconds ---" % (end_time - start_time))
        finalTime=end_time-start_time

        if os.path.exists(output_path+file_name):
            shutil.rmtree(output_path+file_name)
        os.makedirs(output_path+file_name)

        loadedSamples, values_samples, abs_err_samples, rel_err_samples = T.generate_error_samples(finalTime, file_name)
        loadedGolden, values_golden, abs_err_golden, rel_err_golden = T.generate_error_samples(golden_model_time, file_name, golden=True)

        f = open(output_path + file_name + "/" + file_name + "_CDF_summary.out", "w+")
        f.write("Execution Time:"+str(finalTime)+"s \n\n")

        plot_range_analysis_CDF(T.final_quantized_distr, loadedGolden, values_samples, values_golden, f, file_name, range_my_dict.get(file_name))
        plot_error_analysis_CDF(T.abs_err_distr, loadedGolden, abs_err_samples, abs_err_golden, f, file_name, abs_my_dict.get(file_name), rel_my_dict.get(file_name))

        f.flush()
        f.close()

        f = open(output_path + file_name + "/" + file_name + "_PDF_summary.out", "w+")
        f.write("Execution Time:"+str(finalTime)+"s \n\n")

        plot_range_analysis_PDF(T.final_quantized_distr, loadedGolden, values_samples, values_golden, f, file_name, range_my_dict.get(file_name))
        plot_error_analysis_PDF(T.abs_err_distr, loadedGolden, abs_err_samples, abs_err_golden, f, file_name, abs_my_dict.get(file_name), rel_my_dict.get(file_name))
        f.flush()
        f.close()

    except Exception as e:
        logging.error(traceback.format_exc())

    finally:
        del values_samples, abs_err_samples, rel_err_samples
        del values_golden, abs_err_golden, rel_err_golden
        gc.collect()

#error_model.test_HP_error_model()
#error_model.test_LP_error_model()
#error_model.test_typical_error_model()

warnings.warn("Mantissa with implicit bit of sign. In gmpy2 set precision=p includes also sign bit. (e.g. Float32 is mantissa=24 and exp=8)\n")

mantissa=24
exp=8

#xs=[100.0, 115.0, 133.0, 153.0, 176.0, 202.0, 233.0, 268.0,
#    309.0, 356.0, 409.0, 471.0, 543.0, 625.0, 720.0, 829.0, 954.0, 1099.0, 1265.0, 1456.0, 1677.0, 1931.0,
#    2223.0, 2560.0, 2947.0, 3393.0, 3907.0, 4498.0, 5179.0, 5964.0, 6866.0, 7906.0, 9103.0, 10481.0,
#    12068.0, 13895.0, 15999.0, 18421.0, 21210.0, 24421.0, 28118.0, 32375.0, 37276.0, 42919.0, 49417.0,
#    56899.0, 65513.0, 75431.0, 86851.0, 100000.0]

xs=[2,5,10,15,20]

file="/home/roki/GIT/Accuracy/Code/test.txt"

range_my_dict, abs_my_dict, rel_my_dict = getFPTaylorResults(fptaylor_exe, fptaylor_path)
process_file(xs, file, mantissa, exp, range_my_dict, abs_my_dict)

print("\nDone with sample\n")

os._exit(0)
