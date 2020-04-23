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
import numpy

from fpryacc import FPRyacc
from tree_model import TreeModel, copy_tree
from FPTaylor import getFPTaylorResults

import sys
sys.setrecursionlimit(1000000)

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

        loadedSamples, values_samples, abs_err_samples, rel_err_samples = T.generate_error_samples(1, file_name)
        loadedGolden, values_golden, abs_err_golden, rel_err_golden = T.generate_error_samples(golden_model_time, file_name, golden=True)

        f = open(output_path + file_name + "/" + file_name + "_CDF_summary.out", "w+")
        f.write("Execution Time:"+str(finalTime)+"s \n\n")

        plot_range_analysis_CDF(T.final_quantized_distr, loadedGolden, values_samples,
                                values_golden, f, file_name, range_my_dict.get(file_name))

        plot_error_analysis_CDF(T.abs_err_distr, loadedGolden, abs_err_samples,
                                abs_err_golden, f, file_name, abs_my_dict.get(file_name), rel_my_dict.get(file_name), absORrel="Abs")

        plot_error_analysis_CDF(T.relative_err_distr, loadedGolden, rel_err_samples,
                                rel_err_golden, f, file_name, abs_my_dict.get(file_name), rel_my_dict.get(file_name), absORrel="Rel")

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

xs=numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,   14,   17,   21,   24,   27,   31,   34,
         37,   41,   44,   47,   51,   54,   57,   61,   64,   67,   71,
         74,   77,   81,   84,   87,   91,   94,   97,  101,  104,  107,
        111,  114,  117,  121,  124,  127,  131,  134,  137,  141,  144,
        148,  151,  154,  158,  161,  164,  168,  171,  174,  178,  181,
        184,  188,  191,  194,  198,  201,  204,  208,  211,  214,  218,
        221,  224,  228,  231,  234,  238,  241,  244,  248,  251,  254,
        258,  261,  264,  268,  271,  274,  278,  281,  284,  288,  291,
        295,  298,  301,  305,  308,  311,  315,  318,  321,  325,  328,
        331,  335,  338,  341,  345,  348,  351,  355,  358,  361,  365,
        368,  371,  375,  378,  381,  385,  388,  391,  395,  398,  401,
        405,  408,  411,  415,  418,  421,  425,  428,  432,  435,  438,
        442,  445,  448,  452,  455,  458,  462,  465,  468,  472,  475, 478, 480])

file="/home/roki/GIT/paf/test.txt"

range_my_dict, abs_my_dict, rel_my_dict = getFPTaylorResults(fptaylor_exe, fptaylor_path)
process_file(xs, file, mantissa, exp, range_my_dict, abs_my_dict)

print("\nDone with sample\n")

os._exit(0)
