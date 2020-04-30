import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams.update({'font.size': 22})
plt.figure()

x=open("./1_x.txt","r").read().split()
x=list(map(int, x))

for folder in os.listdir("./"):
	if "1_" in folder and ".txt" in folder and not "1_x" in folder:
		tmp=open("./"+folder,"r").read()
		values=list(map(float, tmp.split()))
		plt.plot(x, values, label=folder.split("_")[1], linewidth=3)

plt.title("Dot Product (Variables)")
plt.xlabel('N')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Relative Error')
#plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.legend()
plt.show()
