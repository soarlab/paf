from pacal import *
import matplotlib.pyplot as plt
import numpy as np
import gmpy2
import numpy as np;
from pychebfun import *
from utils import *

from gmpy2 import mpfr
import scipy
import random

##################################
#### How to use FunDistr #########
##################################
# Those need to be defined before PaCAL creates a process pool for
# picklability:
#def example_f1(x):
#    return sqrt(2)/numpy.pi / (1+x**4)
#def example_f2(x):
#    return 1.5*x*x
#f = FunDistr(example_f1 , [-Inf, -1, 0, 1, Inf])
import gmpy2
import math
import pacal
import matplotlib.pyplot as plt
import numpy as np


def computeLargestPositiveNumber(mantissa, exponent):
    with gmpy2.local_context(gmpy2.context(), precision=max(5 * mantissa, 100)) as ctx:
        biggestPositiveNumber = gmpy2.mul(gmpy2.sub(2, gmpy2.exp2(-mantissa)),
                                          gmpy2.exp2(gmpy2.exp2(exponent - 1) - 1))
        # for negative it is just a matter of signs
        return biggestPositiveNumber

class ErrorModelNaive:
    def __init__(self, distribution, precision, samplesize):
        self.inputdistribution=distribution
        self.precision=precision
        self.samplesize=samplesize
        #self.distribution=self.compute_naive_error()

    def compute_naive_error(self):
        x = self.inputdistribution.rand(self.samplesize)
        errors = []
        eps = 2 ** -self.precision
        for r in x:
            #In this way 'e' is always scaled between 0 and 1.
            e = (r - float(gmpy2.round2(r, self.precision))) / (r * eps)
            errors.append(e)
        return x, errors

    def plot_error(self,errors,figureName):
        plt.figure(figureName)
        bin_nb = int(math.ceil(math.sqrt(self.samplesize)))
        n, bins, patches = plt.hist(errors, bins=bin_nb, density=True)
        axes = plt.gca()
        axes.set_xlim([-1, 1])
        return
        # plt.savefig('pics/unifsmall_'+repr(precision))
        # plt.savefig('pics/'+repr(distribution.getName()).replace("'",'')+'_'+repr(precision))
        # plt.clf()



class ErrorModel:

    def __init__(self, wrapperInputDistribution, precision, exp, poly_precision):
        '''
    Constructor interpolates the density function using Chebyshev interpolation
    then uses this interpolation to build a PaCal object:
    the self.distribution attribute which contains all the methods we could possibly want
    Inputs:
        inputdistribution: a PaCal object representing the distribution for which we want to compute
                            the rounding error distribution
        precision, minexp, maxexp: specify the low precision environment suing gmpy2
        poly_precision: the number of exact evaluations of the density function used to
                        build the interpolating polynomial representing it
        '''
        self.wrapperInputDistribution=wrapperInputDistribution
        self.inputdistribution = self.wrapperInputDistribution.execute()

        self.precision=precision
        self.exp=exp
        self.minexp=-(2 ** exp) + 1
        self.maxexp= 2 ** exp

        self.poly_precision=poly_precision
        # Test if the range of floating point number covers enough of the inputdistribution
        x=gmpy2.next_above(gmpy2.inf(-1))
        y=gmpy2.next_below(gmpy2.inf(1))
        coverage=self.inputdistribution.get_piecewise_pdf().integrate(float("-inf"),float("+inf"))
        if coverage<0.99:
            raise Exception('The range of floating points is too narrow, increase maxexp and increase minexp')
        # Builds the Chebyshev polynomial representation of the density function
        self.pdf=chebfun(lambda t:self.__getpdf(t), domain=[-1.0,1.0], N=self.poly_precision)
        #self.pdf2=self.pdf.p
        #plot(self.pdf)
        #plt.show()
        #self.tempPdf=(lambda x : self.pdf(x).item(0))
        #t = self.pdf(0.5).item(0)
        #r = self.pdf(15.5)
        # Creates a PaCal object containing the distribution
        #self.distribution=FunDistr(self.pdf.p, [-1,1])
        self.distribution = FunDistr(self.pdf.p, [-1, 1])
        self.distribution.init_piecewise_pdf()
        print ("ok")

    def setCurrentContextPrecision(self, mantissa, exponent):
        ctx = gmpy2.get_context()
        ctx.precision = mantissa
        ctx.emin = -(2 ** exponent) + 1
        ctx.emax = 2 ** exponent

    def resetContextDefault(self):
        gmpy2.set_context(gmpy2.context())

    # Quick and dirty plotting function
    def plot(self,strFile):
        x=np.linspace(-1,1,201)
        y=self.pdf(x)
        plt.plot(x, y)
        plt.savefig(strFile)
        plt.clf()

    #infVal is finite value
    def getInitialMinValue(self,infVal):
        if not gmpy2.is_finite(infVal):
            print("Error cannot compute intervals with infinity")
            exit(-1)
        bkpCtx = gmpy2.get_context().copy()

        while not gmpy2.is_finite(gmpy2.next_below(infVal)):
            self.setCurrentContextPrecision(self.precision+1,self.exp+1)

        prec= printMPFRExactly(gmpy2.next_below(infVal))
        gmpy2.set_context(bkpCtx)
        return prec

    # infVal is finite value
    def getFinalMaxValue(self, supVal):
        if not gmpy2.is_finite(supVal):
            print("Error cannot compute intervals with infinity")
            exit(-1)
        bkpCtx = gmpy2.get_context().copy()

        while not gmpy2.is_finite(gmpy2.next_above(supVal)):
            self.setCurrentContextPrecision(self.precision + 1, self.exp + 1)

        prec = printMPFRExactly(gmpy2.next_above(supVal))
        gmpy2.set_context(bkpCtx)
        return prec

    # Compute the exact density
    def __getpdf(self, t):
        '''
    Constructs the EXACT probability density function at point t in [-1,1]
    Exact values are used to build the interpolating polynomial
        '''

        self.setCurrentContextPrecision(self.precision, self.exp)
        eps = 2 ** -self.precision

        infVal = mpfr(self.wrapperInputDistribution.a)
        supVal = mpfr(self.wrapperInputDistribution.b)

        if not gmpy2.is_finite(infVal):
            infVal = gmpy2.next_above(infVal)

        if not gmpy2.is_finite(supVal):
            supVal = gmpy2.next_below(supVal)

        sums=[]
        #test if  the input is scalar or an array
        if np.isscalar(t):
            tt=[]
            tt.append(t)
        else:
            tt=t
        # main loop through all floating point numbers in reduced precision
        countInitial=0
        countFinal=0
        for ti in tt:
            sum = 0.0
            err = float(ti)*eps

            x=mpfr(printMPFRExactly(infVal))
            y=gmpy2.next_above(x)
            z=gmpy2.next_above(y)

            xmin = (float(self.getInitialMinValue(infVal))+float(printMPFRExactly(x)))/2
            xmax = (float(printMPFRExactly(x))+float(printMPFRExactly(y)))/2.0
            xp = float(printMPFRExactly(x)) /(1.0-err)
            if xmin < xp < xmax:
                sum+=self.inputdistribution.pdf(xp)*abs(xp)*eps/(1.0-err)
                countInitial=countInitial+1
            # Deal with all standard intervals
            while y<supVal:
                xmin = xmax
                xmax = (float(printMPFRExactly(y))+float(printMPFRExactly(z)))/2.0
                xp = float(printMPFRExactly(y)) / (1.0 - err)

                if xmin < xp < xmax:
                    sum+=self.inputdistribution.pdf(xp)*abs(xp)*eps/(1.0-err)

                y=z
                z=gmpy2.next_above(z)
            # Deal with the very last interval [x,(x+y)/2]
            # Z now should be equal to SUPVAL
            xmin=xmax
            xmax = (float(printMPFRExactly(y)) + float(self.getFinalMaxValue(supVal))) / 2.0
            xp = float(printMPFRExactly(y)) / (1.0 - err)

            #xp=mpfr(str(y))/(1.0-err)
            #xmax = mpfr(str(y))
            if xmin < xp < xmax:
                sum+=self.inputdistribution.pdf(xp)*abs(xp)*eps/(1.0-err)
                countFinal=countFinal+1

            sums.append(sum)

        self.resetContextDefault()

        if np.isscalar(t):
            return sum
        else:
            return sums
