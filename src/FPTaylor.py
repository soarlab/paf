import os
import shlex
import subprocess

def executeOnBenchmarks(fptaylorpath, folder_path, results_folder):
    if os.path.exists(folder_path+results_folder):
        print("WARNING!!! FPTaylor results already computed!")
        return
    else:
        os.makedirs(folder_path+results_folder)
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            print("FpTaylor on: " + str(file))
            exe_line=fptaylorpath+" --rel-error true "+folder_path+file
            exe = shlex.split(exe_line)
            trace = open(folder_path+results_folder+"/"+file, "w+")
            pUNI = subprocess.Popen(exe, shell=False, stdout=trace,stderr=trace)
            pUNI.wait()
    print("Done")

def getAbsoluteError(folder_path):
    my_dict={}
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            f = open(folder_path + "/" + file, "r")
            text = f.readlines()
            file_name = file.split(".")[0]
            value=None
            for line in text:
                if "Absolute error (exact):" in line:
                    value=str(line.split(":")[1])
                    value=value.split("(")[0]
                    value=value.strip()
                    break
            my_dict[file_name.lower()]=value
            my_dict[file_name.lower()+"_gaussian"] = value
            f.close()
    return my_dict
    print("Done")

def getRelativeError(folder_path):
    my_dict={}
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            f = open(folder_path + "/"+ file, "r")
            text = f.readlines()
            file_name = file.split(".")[0]
            value=None
            for line in text:
                if "Relative error (exact):" in line:
                    value=str(line.split(":")[1])
                    break
            my_dict[file_name.lower()]=value
            my_dict[file_name.lower()+"_gaussian"] = value
            f.close()
    return my_dict
    print("Done")

def getBounds(folder_path):
    my_dict={}
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            f = open(folder_path + "/" + file, "r")
            text = f.readlines()
            file_name = file.split(".")[0]
            value=None
            for line in text:
                if "Bounds (floating-point):" in line:
                    value=str(line.split(":")[1])
            my_dict[file_name.lower()]=value
            my_dict[file_name.lower()+"_gaussian"] = value
            f.close()
    return my_dict
    print("Done")

def getFPTaylorResults(fptaylor_command, benchmarks_path):
    results_folder="results"
    executeOnBenchmarks(fptaylor_command, benchmarks_path, results_folder)
    abs_my_dict = getAbsoluteError(benchmarks_path+results_folder)
    rel_my_dict = getRelativeError(benchmarks_path+results_folder)
    range_my_dict = getBounds(benchmarks_path+results_folder)
    if not len(abs_my_dict) == len(rel_my_dict) and not len(range_my_dict) == len(rel_my_dict):
        print("WARNING!!! Mismatch in dictionaries in FPTaylor")
    return range_my_dict, abs_my_dict, rel_my_dict