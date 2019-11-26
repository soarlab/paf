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
    if (coverage < 0.999) or (coverage > 1.001):
        warnings.warn("PDF doesnt integrate to 1. Normalized to integrate to 1.", FutureWarning, stacklevel=2)
        distr_pdf=distr.get_piecewise_pdf()
        distr.piecewise_pdf=(distr_pdf*(1/coverage))
        new_coverage = distr.get_piecewise_pdf().integrate(float("-inf"), float("+inf"))
        print(new_coverage)
    return distr
def getBoundsWhenOutOfRange(distribution, mantissa, exponent):
    minVal = min(distribution.get_piecewise_pdf().breaks)
    maxVal = max(distribution.get_piecewise_pdf().breaks)
    res=checkBoundsOutOfRange(minVal, maxVal,mantissa,exponent)
    return res