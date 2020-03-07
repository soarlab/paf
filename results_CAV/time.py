import os
import numpy as np

for folder in os.listdir("./"):
	if not "__pycache__" in folder and "gaussian" in folder and not ".py" in folder:
		print(folder)
		cdf_distances=open("./"+folder+"/"+folder+"_CDF_summary.out","r").readlines()
		for line in cdf_distances:
			if "Execution Time" in line:
				print(line)
