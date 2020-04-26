import matplotlib.pyplot as plt
import os
import numpy as np

for folder in os.listdir("./"):
	if "3_a_" in folder and ".txt" in folder:
		print(folder)
		n=(folder.split("_")[2])
		m=(folder.split("_")[3]).split(".")[0]
		
		a_file=open("./"+folder,"r")
		a=a_file.read().strip()
		
		b_file=open("./3_b_"+n+"_"+m+".txt","r")
		b=b_file.read().strip()
		
		a_file.close()
		b_file.close()
		
		init="fake:U(0,1)\n"
		expr="("+a+"*"+b+")"
		for i in range(1,int(n)):
			expr="("+expr+"+("+str(a)+"*"+str(b)+")"+")"
					
		out=open("./results/n_"+str(n)+"_i_"+str(m)+".txt","w+")
		out.write(init+expr)
		out.close()
