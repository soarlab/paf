import os
import shlex
import shutil
import subprocess

def executeOnBenchmarks(fptaylorpath, folder_path):
    if os.path.exists(folder_path+"/results"):
        print("WARNING!!! FPTaylor results already computed!")
        return
    else:
        os.makedirs(folder_path+"/results")
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            print("FpTaylor on: " + str(file))
            exe_line=fptaylorpath+" --rel-error true "+folder_path+file
            exe = shlex.split(exe_line)
            trace = open(folder_path+"results/"+file, "w+")
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
                    break
            my_dict[file_name.lower()]=value
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
            f.close()
    return my_dict
    print("Done")