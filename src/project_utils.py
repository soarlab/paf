import warnings
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN

import gmpy2
import numpy
import pacal

from setup_utils import digits_for_range


def round_number_up_to_digits(number, digits):
    return ("{0:."+str(digits)+"Uf}").format(number)

def round_number_down_to_digits(number, digits):
    return ("{0:."+str(digits)+"Df}").format(number)

def round_number_nearest_to_digits(number, digits):
    return ("{0:."+str(digits)+"Nf}").format(number)

def round_down(number, digits):
    return Decimal(number).quantize(Decimal("."+("".zfill(digits))), rounding=ROUND_FLOOR)

def round_up(number, digits):
    return Decimal(number).quantize(Decimal("."+("".zfill(digits))), rounding=ROUND_CEILING)

def round_near(number, digits):
    return Decimal(number).quantize(Decimal("."+("".zfill(digits))), rounding=ROUND_HALF_EVEN)

def dec2Str(dec):
    return '{0:f}'.format(dec)

def set_context_precision(mantissa, exponent):
    ctx = gmpy2.get_context()
    #precision is already +1, meaning in single precision is 24 (and not 23)
    ctx.precision = mantissa
    ctx.emax = 2 ** (exponent - 1)
    # 4-emax-precision
    #https://github.com/aleaxit/gmpy/issues/103
    ctx.emin = 4-ctx.emax-ctx.precision
    return ctx

def reset_default_precision():
    gmpy2.set_context(gmpy2.context())

def isNumeric(n):
    try:
        float(n)
        return True
    except:
        return False

class MyFunDistr(pacal.FunDistr):
    """General distribution defined as function with
    singularities at given breakPoints."""
    def __init__(self, name, interpolator, breakPoints = None, interpolated = False):
        self.name=name
        super(MyFunDistr, self).__init__(interpolator, breakPoints=breakPoints, interpolated=interpolated)

    def rand_raw(self, n = 1):
        y = numpy.random.uniform(0, 1, n)
        tmp = self.get_piecewise_invcdf(use_interpolated=True)(y)
        if numpy.isnan(tmp).any():
            invFun = self.get_piecewise_cdf_interp().invfun(use_interpolated=False)
            tmp = numpy.zeros(len(y))
            for index, val in enumerate(y):
                tmp[index] = invFun(val)
        return tmp

    def getName(self):
        return self.name

def printMPFRExactly(a):
    return "{0:.100f}".format(a)

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

def remove_starting_minus_from_string(string):
    if string[0]=="-":
        return string[1:]
    else:
        print("This method should be used only by the parser. Input string does not start with -")
        exit(-1)


def linear_space_with_decimals(low, up, inc_low, inc_up, n):
    vals=numpy.linspace(float(low), float(up), endpoint=True, num=n+1)
    if vals[0]==vals[-1]:
        return []
    tmp=[]
    ret=[]
    for val in vals:
        if not (round_near(Decimal(val), digits_for_range), True) in tmp:
            tmp.append((round_near(Decimal(val), digits_for_range), True))
    tmp[0]= (Decimal(low), inc_low)
    tmp[-1]=(Decimal(up), inc_up)
    for ind, val in enumerate(tmp[:-1]):
        if val[0]>=tmp[ind+1][0]:
            print("\n\nAccuracy problem with interval linspace\n\n")
            #exit(-1)
    for ind, val in enumerate(tmp[:-1]):
        ret.append([val,tmp[ind+1],False])
    return ret