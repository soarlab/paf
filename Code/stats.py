import gmpy2
import math
import pacal
import matplotlib.pyplot as plt
import numpy as np

def plot_error(distribution, precision, samplesize):
    x=distribution.rand(samplesize)
    errors=[]
    bin_nb=math.ceil(math.sqrt(samplesize))
    print('Number of bins='+ repr(bin_nb))
    for r in x:
        e=float(r-gmpy2.round2(r,precision))
        errors.append(e)
    n, bins, patches =  plt.hist(errors,bins=bin_nb, density=True)
    plt.show()
