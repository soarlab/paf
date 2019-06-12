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
    eps=2**-precision
    for r in x:
        e=(r-float(gmpy2.round2(r,precision)))/(r*eps)
        errors.append(e)
    n, bins, patches =  plt.hist(errors,bins=bin_nb, density=True)
    axes = plt.gca()
    axes.set_xlim([-1,1])
    plt.savefig('pics/reallyweird_'+repr(precision))
    #plt.savefig('pics/'+repr(distribution.getName()).replace("'",'')+'_'+repr(precision))
    plt.clf()
