import warnings

import gmpy2
import matplotlib.pyplot as plt
import numpy
import pacal
from pacal import *


def set_context_precision(mantissa, exponent):
    ctx = gmpy2.get_context()
    ctx.precision = mantissa
    ctx.emax = 2 ** (exponent - 1)
    ctx.emin = 1 - ctx.emax


def reset_default_precision():
    gmpy2.set_context(gmpy2.context())


def setCurrentContextPrecision(mantissa, exponent):
    ctx = gmpy2.get_context()
    ctx.precision = mantissa
    ctx.emax = 2 ** (exponent - 1)
    ctx.emin = 1 - ctx.emax


def resetContextDefault():
    gmpy2.set_context(gmpy2.context())


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
    else:
        warnings.warn("PDF integrates to 1. Good Accuracy", FutureWarning, stacklevel=2)
    return distr

def getBoundsWhenOutOfRange(distribution, mantissa, exponent):
    minVal = min(distribution.get_piecewise_pdf().breaks)
    maxVal = max(distribution.get_piecewise_pdf().breaks)
    res=checkBoundsOutOfRange(minVal, maxVal,mantissa,exponent)
    return res
