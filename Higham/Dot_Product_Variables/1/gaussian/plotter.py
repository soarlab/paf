import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams.update({'font.size': 35})
plt.figure(figsize=(17, 10))

x=open("./1_x.txt","r").read().split()
x=list(map(int, x))

markers=[".","v","p","s","x"]
i=0

#lambda_="1"
#for folder in os.listdir("./"):
#	if "1_" in folder and ".txt" in folder and not "1_x" in folder:

tmp=open("./1_detbound.txt","r").read()
values=list(map(float, tmp.split()))
plt.plot(x, values, label="detbound", linewidth=3, marker=".", markersize=30)

tmp=open("./1_probound_lambda_5.txt","r").read()
values=list(map(float, tmp.split()))
plt.plot(x, values, label="probound-lambda-5", linewidth=3, marker="^", markersize=20)

tmp=open("./1_probound_lambda_1.txt","r").read()
values=list(map(float, tmp.split()))
plt.plot(x, values, label="probound-lambda-1", linewidth=3, marker="v", markersize=20)

tmp=open("./1_paf.txt","r").read()
values=list(map(float, tmp.split()))
plt.plot(x, values, label="paf", linewidth=3, marker="p", markersize=20)

tmp=open("./1_errmax.txt","r").read()
values=list(map(float, tmp.split()))
plt.plot(x, values, label="errmax", linewidth=3, marker="s", markersize=15)

tmp=open("./1_erravg.txt","r").read()
values=list(map(float, tmp.split()))
plt.plot(x, values, label="erravg", linewidth=3, marker="X", markersize=20)

#plt.title("inner product (random entries)")
plt.xlabel('n')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('relative error')
#plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.legend(frameon=False, bbox_to_anchor=(-0.02,1.04), loc="upper left", labelspacing=.3)

plt.savefig("normal_lambda_1_and_5.png", bbox_inches = 'tight',
    pad_inches = 0)

plt.show()
