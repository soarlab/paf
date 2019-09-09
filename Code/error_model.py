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

class ErrorModelNaive:
    def __init__(self, distribution, precision, samplesize):
        self.inputdistribution=distribution
        self.precision=precision
        self.samplesize=samplesize
        self.distribution=self.compute_naive_error()

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

    def __init__(self, wrapperInputDistribution, precision, minexp, maxexp, poly_precision):
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
        self.minexp=minexp
        self.maxexp=maxexp
        self.poly_precision=poly_precision
        # Test if the range of floating point number covers enough of the inputdistribution
        x=gmpy2.next_above(gmpy2.inf(-1))
        y=gmpy2.next_below(gmpy2.inf(1))
        coverage=self.inputdistribution.get_piecewise_pdf().integrate(float("-inf"),float("+inf"))
        #if (1.0-coverage)>0.001:
        #    raise Exception('The range of floating points is too narrow, increase maxexp and increase minexp')
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


        # Quick and dirty plotting function
    def plot(self,strFile):
        x=np.linspace(-1,1,201)
        y=self.pdf(x)
        plt.plot(x, y)
        plt.savefig(strFile)
        plt.clf()


        # Compute the exact density
    def __getpdf(self, t):
        '''
    Constructs the EXACT probability density function at point t in [-1,1]
    Exact values are used to build the interpolating polynomial
        '''

        #with gmpy2.ieee(64) as ctx:
        infVal = mpfr(self.wrapperInputDistribution.a)
        supVal = mpfr(self.wrapperInputDistribution.b)
        if not gmpy2.is_finite(infVal):
            infVal = gmpy2.next_above(infVal)
        if not gmpy2.is_finite(supVal):
            supVal = gmpy2.next_above(supVal)

        eps = 2 ** -self.precision

        ctx = gmpy2.get_context()
        ctx.precision=self.precision
        ctx.emin=self.minexp
        ctx.emax=self.maxexp

        sums=[]
        #test if  the input is scalar or an array
        if np.isscalar(t):
            tt=[]
            tt.append(t)
        else:
            tt=t
        # main loop through all floating point numbers in reduced precision
        for ti in tt:
            sum = 0.0
            err = float(ti)*eps
            #x=gmpy2.next_above(infVal)

            x=mpfr(printMPFRExactly(infVal))
            y=gmpy2.next_above(x)
            z=gmpy2.next_above(y)

            xmin = float(printMPFRExactly(x))
            xmax = (xmin+float(printMPFRExactly(y)))/2.0
            xp = xmin/(1.0-err)
            if xmin < xp < xmax:
                sum+=self.inputdistribution.pdf(xp)*abs(xp)*eps/(1.0-err)
            # Deal with all standard intervals
            while z<supVal:
                #ctx.precision=53
                #ctx.emin=-1023
                #ctx.emax=1023
                xmin = xmax
                xmax = (float(printMPFRExactly(y))+float(printMPFRExactly(z)))/2.0
                xp = float(printMPFRExactly(y)) / (1.0 - err)
                #xmax=(float(y) + float(z))/2.0
                #xp = (xmin + xmax)/2.0

                if xmin < xp < xmax:
                    sum+=self.inputdistribution.pdf(xp)*abs(xp)*eps/(1.0-err)

                #ctx.precision=self.precision
                #ctx.emin=self.minexp
                #ctx.emax=self.maxexp

                x=y
                y=z
                z=gmpy2.next_above(z)
            # Deal with the very last interval [x,(x+y)/2]
            xmin=xmax
            xmax = (float(printMPFRExactly(y)) + float(printMPFRExactly(z))) / 2.0
            xp = float(printMPFRExactly(y)) / (1.0 - err)

            #xp=mpfr(str(y))/(1.0-err)
            #xmax = mpfr(str(y))
            if xmin <= xp <= xmax:
                sum+=self.inputdistribution.pdf(xp)*abs(xp)*eps/(1.0-err)
            #print('Evaluated at '+repr(ti)+'    Result='+repr(sum))
            sums.append(sum)
        if np.isscalar(t):
            return sum
        else:
            return sums
