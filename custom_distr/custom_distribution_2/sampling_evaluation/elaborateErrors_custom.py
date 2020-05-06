import os
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np

quartiles=["50.0%"]

golden_list=[]
sampling_list=[]
paf_list=[]

for folder in os.listdir("./"):
	print(folder)
	
	if "custom_distribution_2" in folder:		

		print("\n\n\n\n\n"+folder)
		
		try:
				
			golden=open("./"+folder+"/golden.txt","r").read()
			sampling=open("./"+folder+"/sampling.txt","r").read()
			FPPM=open("./"+folder+"/"+folder+"_CDF_summary.out","r").read()
			lookFor="CDF Range Analysis"
			
			for quart in quartiles:
				
				golden_range=(golden.split(lookFor)[1]).split("##################################")[0]
				sampling_range=(sampling.split(lookFor)[1]).split("##################################")[0]
				FPPM_range=(FPPM.split(lookFor)[1]).split("##################################")[0]
				
				for line in FPPM_range.split("\n"):
					if quart in line:
						vals=float((((line.split("[")[1]).split("]")[0]).split(","))[1])
						#print(("{:.7e}".format(FPPM_vals)))
						paf_list.append(float("{:.7e}".format(vals)))
				
				for line in golden_range.split("\n"):
					if quart in line:
						vals=float((((line.split("[")[1]).split("]")[0]).split(","))[1])
						#print(("{:.7e}".format(FPPM_vals)))
						golden_list.append(float("{:.7e}".format(vals)))
						
				for line in sampling_range.split("\n"):
					if quart in line:
						vals=float((((line.split("[")[1]).split("]")[0]).split(","))[1])
						#print(("{:.7e}".format(FPPM_vals)))
						sampling_list.append(float("{:.7e}".format(vals)))
		except:
			print("Not yet done")

print(golden_list)
print(set(paf_list))
print(set(golden_list))

plt.rcParams.update({'font.size': 22})
plt.figure()

plt.hist(sampling_list, bins=10, linewidth=3)
plt.hist(golden_list, bins=100, linewidth=1)
plt.hist(paf_list, bins=100, linewidth=1, color="red")

#plt.title("Dot Product")
#plt.xlabel('N')
#plt.ylabel('Relative Error')
#plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
#plt.legend()
plt.show()

