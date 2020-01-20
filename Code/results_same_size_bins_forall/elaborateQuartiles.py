import os
import numpy as np

bins=10000
quartiles=["25.0","50.0","75.0","100"]
print("Bins"+str(bins))
for folder in os.listdir("./"):
	if not "__" in folder:
		print(folder)
		golden=open("./"+folder+"/golden.txt","r").readlines()
		sampling=open("./"+folder+"/sampling.txt","r").readlines()
		FPPM=open("./"+folder+"/"+folder+"_summary.out","r").readlines()
		for indGolden,lineGolden in enumerate(golden):
			if "###### Info" in lineGolden and (str(bins)+"#") in lineGolden:
				i=1
				j=0
				goldenValues=[]
				while j<len(quartiles):
					if str(quartiles[j]) in str(golden[indGolden+i]):
						goldenValues.append(((golden[indGolden+i].split("[")[1]).split("]")[0]).split(","))
						j=j+1
					i=i+1
		for indSampling, lineSampling in enumerate(sampling):
			if "###### Info" in lineSampling and (str(bins)+"#") in lineSampling:
				i=1
				j=0
				samplingValues=[]
				while j<len(quartiles):
					if quartiles[j] in sampling[indSampling+i]:
						samplingValues.append(((sampling[indSampling+i].split("[")[1]).split("]")[0]).split(","))
						j=j+1
					i=i+1	
		for indFPPM, lineFPPM in enumerate(FPPM):
			if "###### Info" in lineFPPM and (str(bins)+" bins") in lineFPPM:
				i=1
				j=0
				FPPMValues=[]
				while j<len(quartiles):
					if quartiles[j] in FPPM[indFPPM+i]:
						FPPMValues.append(((FPPM[indFPPM+i].split("[")[1]).split("]")[0]).split(","))
						j=j+1
					i=i+1
		
		for ind,val in enumerate(quartiles):
			print("Quartile"+str(val))
			g = goldenValues[ind]
			s = samplingValues[ind]
			m = FPPMValues[ind]
			print(g,s,m)
			print("Gap Golden -> Sampling"+str(abs(float(g[0])-float(s[0])))+","+str(abs(float(g[1])-float(s[1]))))
			print("Gap Golden -> Model"+str(abs(float(g[0])-float(m[0])))+","+str(abs(float(g[1])-float(m[1]))))
			input()
