import gmpy2
import warnings
import matplotlib.pyplot as plt
import pacal
from pacal import *
import numpy
from gmpy2 import *

def init_pacal():
    '''Limit pacal threads'''
    pacal.params.general.nprocs = 8
    pacal.params.general.parallel = True

class MyFunDistr(pacal.FunDistr):
    """General distribution defined as function with
    singularities at given breakPoints."""
    def rand_raw(self, n = 1):
        y = numpy.random.uniform(0, 1, n)
        tmp=self.get_piecewise_invcdf(use_interpolated=True)(y)
        if numpy.isnan(tmp).any():
            invFun=self.get_piecewise_cdf_interp().invfun(use_interpolated=False)
            tmp=numpy.zeros(len(y))
            for index, val in enumerate(y):
                tmp[index]=invFun(val)
        return tmp

def plotTicks(figureName, mark, col, lw, s, ticks, label=""):
    if not ticks is None:
        values_ticks=eval(ticks)
        minVal = values_ticks[0]
        maxVal = values_ticks[1]
        labelMinVal = str("%.7f" % float(minVal))
        labelMaxVal = str("%.7f" % float(maxVal))
        plt.figure(figureName)
        plt.scatter(x=[minVal, maxVal], y=[0, 0], c=col, marker=mark, label="FPT: [" + labelMinVal + "," + labelMaxVal + "]", linewidth=lw, s=s)


def plotBoundsDistr(figureName, distribution):
    minVal = distribution.range_()[0]
    maxVal = distribution.range_()[1]
    labelMinVal = str("%.7f" % distribution.range_()[0])
    labelMaxVal = str("%.7f" % distribution.range_()[1])
    plt.figure(figureName)
    plt.scatter(x=[minVal, maxVal], y=[0, 0], c='r', marker="|",
                label="PM: [" + labelMinVal + "," + labelMaxVal + "]", linewidth=6, s=600)

def printMPFRExactly(a):
    return "{0:.50f}".format(a)

def computeLargestPositiveNumber(mantissa, exponent):
    assert "mantissa includes sign bit!"
    with gmpy2.local_context(gmpy2.context(), precision=100) as ctx:
        biggestPositiveNumber = gmpy2.mul(gmpy2.sub(2, gmpy2.exp2(-(mantissa-1))),
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

def normalizeDistribution(distr, init=False):
    coverage = distr.get_piecewise_pdf().integrate(float("-inf"), float("+inf"))
    if abs(coverage)<0.1 or abs(coverage)>2.0:
        if not init:
            print("Coverage warning: accuracy problem with Pacal. Normalization not done.")
            return distr
    if (coverage < 0.99999) or (coverage > 1.00001):
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