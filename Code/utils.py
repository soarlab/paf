import gmpy2
import warnings
import matplotlib.pyplot as plt
from pacal import *
from gmpy2 import *

def printMPFRExactly(a):
    return "{0:.50f}".format(a)

def computeLargestPositiveNumber(mantissa, exponent):
    with gmpy2.local_context(gmpy2.context(), precision=max(5 * mantissa, 100)) as ctx:
        biggestPositiveNumber = gmpy2.mul(gmpy2.sub(2, gmpy2.exp2(-mantissa)),
                                          gmpy2.exp2(gmpy2.exp2(exponent - 1) - 1))
        # for negative it is just a matter of signs
        return biggestPositiveNumber

def checkBoundsOutOfRange(a, b, mantissa, exponent):
    ret=[0,0]
    val=float(printMPFRExactly(computeLargestPositiveNumber(mantissa, exponent)))
    val=val+1
    if a<val<b:
        warnings.warn("The range of floating points is too narrow, increase exponent", FutureWarning, stacklevel=2)
        ret[1]=val
    if a<-val<b:
        warnings.warn("The range of floating points is too narrow, increase exponent", FutureWarning, stacklevel=2)
        ret[0]=val
    return ret

def normalizeDistribution(distr):
    coverage = distr.get_piecewise_pdf().integrate(float("-inf"), float("+inf"))

    tmp = FunDistr(distr.get_piecewise_pdf, distr.breakPoints, interpolated=True)
    tmp.init_piecewise_pdf()
    val=tmp.get_piecewise_pdf().integrate(float("-inf"), float("+inf"))
    #self.distribution.init_piecewise_pdf()

    if coverage < 0.999:
        warnings.warn("PDF doesnt integrate to 1. Normalized to integrate to 1.", FutureWarning, stacklevel=2)
        ret=distr*(1.0/coverage)
        ret._1_scale = 1.0
        ret.init_piecewise_pdf()
        print(ret.get_piecewise_pdf().integrate(float("-inf"), float("+inf")))
        plt.figure("tmp")
        ret.plot()
        print(ret.get_piecewise_pdf().integrate(float("-inf"), float("+inf")))
        return ret
    else:
        return distr