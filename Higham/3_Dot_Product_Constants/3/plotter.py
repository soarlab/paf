import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams.update({'font.size': 33})
plt.figure(figsize=(17, 10))

x=open("./3_x.txt","r").read().split()
x=list(map(int, x))

markers=[".","v","p","s","x"]
i=0

#for folder in os.listdir("./"):
#	if "1_" in folder and ".txt" in folder and not "1_x" in folder:

tmp=open("./3_detbound.txt","r").read()
values=list(map(float, tmp.split()))
plt.plot(x, values, label="detbound", linewidth=3, marker=".", markersize=30)

tmp=open("./3_probound.txt","r").read()
values=list(map(float, tmp.split()))
plt.plot(x, values, label="probound", linewidth=3, marker="v", markersize=20)

tmp=open("./3_paf.txt","r").read()
values=list(map(float, tmp.split()))
plt.plot(x, values, label="paf", linewidth=3, marker="p", markersize=20)

tmp=open("./3_errmax.txt","r").read()
values=list(map(float, tmp.split()))
plt.plot(x, values, label="errmax", linewidth=3, marker="s", markersize=15)

tmp=open("./3_erravg.txt","r").read()
values=list(map(float, tmp.split()))
plt.plot(x, values, label="erravg", linewidth=3, marker="X", markersize=20)

#plt.title("inner product (random entries)")
plt.xlabel('n')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('relative error')
#plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.legend(frameon=False)

plt.savefig("constants.png", bbox_inches = 'tight',
    pad_inches = 0)

plt.show()
