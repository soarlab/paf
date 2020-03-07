import os
import numpy as np

for folder in os.listdir("./"):
	if not "__pycache__" in folder and not "gaussian" in folder and not ".py" in folder:
		print(folder)
		cdf_distances=open("./"+folder+"/"+folder+"_CDF_summary.out","r").read()
		my_range=(cdf_distances.split("CDF Measure Distances Range Analysis")[1]).split("##################################")[0]
		
		final=""
		for line in my_range.split("\n"):
			if not line.isspace():
				if "Variational Distance - Golden -> DistrPM" in line:
					final=final+ " & " + '%.4f' % float(line.split(":")[1])
					continue
				if "Variational Distance - Golden -> Sampling" in line:
					final=final+ " & " + '%.4f' % float(line.split(":")[1])
					continue
				if "AVG Variational Distance - Golden -> DistrPM" in line:
					final=final+ " & " + '%.4f' % float(line.split(":")[1])
					continue
				if "AVG Variational Distance - Golden -> Sampling" in line:
					final=final+ " & " + '%.4f' % float(line.split(":")[1])
					continue
		
		my_range=(cdf_distances.split("Measure Distances Abs Error Analysis")[1]).split("##################################")[0]
		
		for line in my_range.split("\n"):
			if not line.isspace():
				if "Variational Distance - Golden -> DistrPM" in line:
					final=final+ " & " + '%.4f' % float(line.split(":")[1])
					continue
				if "Variational Distance - Golden -> Sampling" in line:
					final=final+ " & " + '%.4f' % float(line.split(":")[1])
					continue
				if "AVG Variational Distance - Golden -> DistrPM" in line:
					final=final+ " & " + '%.4f' % float(line.split(":")[1])
					continue
				if "AVG Variational Distance - Golden -> Sampling" in line:
					final=final+ " & " + '%.4f' % float(line.split(":")[1])
					continue
		print(final)
