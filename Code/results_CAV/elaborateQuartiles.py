import os
import numpy as np

quartiles=["99.99%"]
fptaylor=open("./rangefpt.py.txt","r").read()

for folder in os.listdir("./"):
	if not "__pycache__" in folder and not "gaussian" in folder and not ".py" in folder:
		print("\n\n\n\n\n"+folder)
		
		golden=open("./"+folder+"/golden.txt","r").read()
		sampling=open("./"+folder+"/sampling.txt","r").read()
		FPPM=open("./"+folder+"/"+folder+"_PDF_summary.out","r").read()
		
		golden_g=open("./"+folder+"_gaussian"+"/golden.txt","r").read()
		sampling_g=open("./"+folder+"_gaussian"+"/sampling.txt","r").read()
		FPPM_g=open("./"+folder+"_gaussian"+"/"+folder+"_gaussian"+"_PDF_summary.out","r").read()
		
		lookFor="PDF Range Analysis"
		
		for line in fptaylor.split("\n"):
			if folder in line:
				fptaylor_vals=[float("{0:.2f}".format(float(i))) for i in ((line.split("[")[1]).split("]")[0]).split(",")]
				fptaylor_range=abs(float(fptaylor_vals[0])-float(fptaylor_vals[1]))
				print("FPtaylor: "+str(fptaylor_vals)+"\n")
				break
		
		for quart in quartiles:
			
			golden_range=(golden.split(lookFor)[1]).split("##################################")[0]
			sampling_range=(sampling.split(lookFor)[1]).split("##################################")[0]
			FPPM_range=(FPPM.split(lookFor)[1]).split("##################################")[0]
			
			for line in FPPM_range.split("\n"):
				if quart in line:
					FPPM_vals=[float("{0:.2f}".format(float(i))) for i in (((line.split("[")[1]).split("]")[0]).split(","))]
					FPPM_range=abs(float(FPPM_vals[0])-float(FPPM_vals[1]))
					print("FPPM uni: "+str(FPPM_vals)+","+str(FPPM_range))


			for line in golden_range.split("\n"):
				if quart in line:
					golden_vals=[float("{0:.2f}".format(float(i))) for i in ((line.split("[")[1]).split("]")[0]).split(",")]
					golden_range=abs(float(golden_vals[0])-float(golden_vals[1]))
					print("golden uni: "+str(golden_vals)+","+str(golden_range))

					 
			for line in sampling_range.split("\n"):
				if quart in line:
					sampling_vals=[float("{0:.2f}".format(float(i))) for i in ((line.split("[")[1]).split("]")[0]).split(",")]
					sampling_range=abs(float(sampling_vals[0])-float(sampling_vals[1]))
					print("sampling uni: "+str(sampling_vals)+","+str(sampling_range)+"\n\n")
			
			golden_range_g=(golden_g.split(lookFor)[1]).split("##################################")[0]
			sampling_range_g=(sampling_g.split(lookFor)[1]).split("##################################")[0]
			FPPM_range_g=(FPPM_g.split(lookFor)[1]).split("##################################")[0]
			
			for line in FPPM_range_g.split("\n"):
				if quart in line:
					FPPM_vals_g=[float("{0:.2f}".format(float(i))) for i in ((line.split("[")[1]).split("]")[0]).split(",")]
					FPPM_range_g=abs(float(FPPM_vals_g[0])-float(FPPM_vals_g[1]))
					print("FPPM gau: "+str(FPPM_vals_g)+","+str(FPPM_range_g))

			for line in golden_range_g.split("\n"):
				if quart in line:
					golden_vals_g=[float("{0:.2f}".format(float(i))) for i in ((line.split("[")[1]).split("]")[0]).split(",")]
					golden_range_g=abs(float(golden_vals_g[0])-float(golden_vals_g[1]))
					print("Golden gau: "+str(golden_vals_g)+","+str(golden_range_g))
					 
			for line in sampling_range_g.split("\n"):
				if quart in line:
					sampling_vals_g=[float("{0:.2f}".format(float(i))) for i in ((line.split("[")[1]).split("]")[0]).split(",")]
					sampling_range_g=abs(float(sampling_vals_g[0])-float(sampling_vals_g[1]))
					print("Sampling gau: "+str(sampling_vals_g)+","+str(sampling_range_g)+"\n\n")
			
			print ("Max Fptaylor is: "+str(fptaylor_range))
			print ("Max Uni is: "+str((FPPM_range)))
			max_uni="{0:.2f}".format((FPPM_range)/fptaylor_range)
			print ("Max Gau is: "+str((FPPM_range_g)))
			max_gau="{0:.2f}".format((FPPM_range_g)/fptaylor_range)
			print (FPPM_vals[0],"&",FPPM_vals[1],"&", FPPM_vals_g[0], "&", FPPM_vals_g[1], "&", fptaylor_vals[0], "&", fptaylor_vals[1], "&", max_uni, "&", max_gau, "\\\\") 
			
		input()
