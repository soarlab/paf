import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams.update({'font.size': 22})
plt.figure()

x=open("./3_x.txt","r").read().split()
x=list(map(int, x))

markers=[".","v","p","s","x"]
i=0
for folder in os.listdir("./"):
	print(folder)
	if "3_" in folder and ".txt" in folder and not "3_x" in folder:
		tmp=open("./"+folder,"r").read()
		values=list(map(float, tmp.split()))
		#print(folder)
		#print(values)
		plt.plot(x, values, label=folder.split("_")[1], linewidth=2, marker=markers[i], markersize=12)
		i=i+1

plt.title("inner product (constant entries)")
plt.xlabel('n')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('relative error')
#plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.legend()
plt.show()
		
