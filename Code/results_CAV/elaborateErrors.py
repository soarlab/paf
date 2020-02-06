import os
import numpy as np

quartiles=["99.99%"]
fptaylor=open("./absfpt.py.txt","r").read()

for folder in os.listdir("./"):
	if not "__pycache__" in folder and "gaussian" in folder and not ".py" in folder:
		print("\n\n\n\n\n"+folder)
		
		golden=open("./"+folder+"/golden.txt","r").read()
		sampling=open("./"+folder+"/sampling.txt","r").read()
		FPPM=open("./"+folder+"/"+folder+"_PDF_summary.out","r").read()
			
		lookFor="PDF Error Analysis"
		
		for line in fptaylor.split("\n"):
			if folder in line:
				fptaylor_vals=float(line.split()[1]) 
				print("FPtaylor: "+("{:.2e}".format(fptaylor_vals))+"\n")
				break
		
		for quart in quartiles:
			
			golden_range=(golden.split(lookFor)[1]).split("##################################")[0]
			sampling_range=(sampling.split(lookFor)[1]).split("##################################")[0]
			FPPM_range=(FPPM.split(lookFor)[1]).split("##################################")[0]
			
			for line in FPPM_range.split("\n"):
				if quart in line:
					FPPM_vals=float((((line.split("[")[1]).split("]")[0]).split(","))[1])
					print("FPPM uni: "+("{:.2e}".format(FPPM_vals)))
