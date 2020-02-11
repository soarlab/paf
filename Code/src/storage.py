import numpy as np
import os
from setup_utils import storage_path, storeIfDoesnExist

def load_histograms_range_from_disk(file_name):
    data_auto = np.load(storage_path + file_name + "/" + file_name + "_range_auto.npz")
    data_10000 = np.load(storage_path + file_name + "/" + file_name + "_range_10000.npz")
    notnorm_vals_golden, notnorm_edges_golden = data_auto["vals_golden"], data_auto["edges_golden"]
    data_auto.close()
    data_10000.close()
    return notnorm_vals_golden, notnorm_edges_golden

def load_histograms_error_from_disk(file_name):
    data_auto = np.load(storage_path + file_name + "/" + file_name + "_error_auto.npz")
    data_10000 = np.load(storage_path + file_name + "/" + file_name + "_error_10000.npz")
    notnorm_vals_golden, notnorm_edges_golden = data_auto["vals_golden"], data_auto["edges_golden"]
    data_auto.close()
    data_10000.close()
    return notnorm_vals_golden, notnorm_edges_golden

def store_histograms_range(file_name, vals_golden_auto, edges_golden_auto, vals_golden_10000, edges_golden_10000):
    if storeIfDoesnExist:
        if not os.path.exists(storage_path + file_name + "/"):
            os.makedirs(storage_path + file_name + "/")
            np.savez(storage_path + file_name + "/" + file_name + "_range_10000.npz", vals_golden=vals_golden_10000,
                     edges_golden=edges_golden_10000)
            np.savez(storage_path + file_name + "/" + file_name + "_range_auto.npz", vals_golden=vals_golden_auto,
                     edges_golden=edges_golden_auto)
            print("Store succeed!")
        else:
            print("Golden distribution already exists on the disk!(skip storage)")
    else:
        print("Storage flag set to False (skip storage)")

def store_histograms_error(file_name, vals_golden_auto, edges_golden_auto, vals_golden_10000, edges_golden_10000):
    if storeIfDoesnExist:
        if not os.path.exists(storage_path + file_name + "/"):
            os.makedirs(storage_path + file_name + "/")
            np.savez(storage_path + file_name + "/" + file_name + "_error_10000.npz", vals_golden=vals_golden_10000,
                     edges_golden=edges_golden_10000)
            np.savez(storage_path + file_name + "/" + file_name + "_error_auto.npz", vals_golden=vals_golden_auto,
                     edges_golden=edges_golden_auto)
            print("Store succeed!")
        else:
            print("Golden distribution already exists on the disk!(skip storage)")
    else:
        print("Storage flag set to False (skip storage)")