from pacal import FunDistr
import matplotlib.pyplot as plt
import numpy as np
from pychebfun import Chebfun
import gmpy2
from gmpy2 import mpfr
import random


class ErrorModel:

    def __init__(self, inputdistribution, precision, minexp, maxexp, poly_precision):
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
        self.inputdistribution = inputdistribution
        self.precision=precision
        self.minexp=minexp
        self.maxexp=maxexp
        self.poly_precision=poly_precision
        # Test if the range of floating point number covers enough of the inputdistribution
        x=gmpy2.next_above(gmpy2.inf(-1))
        y=gmpy2.next_below(gmpy2.inf(1))
        coverage=self.inputdistribution.get_piecewise_pdf().integrate(float(x),float(y))
        if (1.0-coverage)>0.001:
            raise Exception('The range of floating points is too narrow, increase maxexp and increase minexp')
        # Builds the Chebyshev polynomial rerpesentation of the density function
        self.pdf=Chebfun.from_function(lambda t:self.__getpdf(t),N=self.poly_precision)
        # Creates a PaCal object containing the distribution
        self.distribution=FunDistr(self.pdf, [-1,1])


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
        ctx=gmpy2.get_context()
        ctx.precision=self.precision
        ctx.emin=self.minexp
        ctx.emax=self.maxexp
        eps=2**-self.precision
        sums=[]
        #test if  the input is scalar or an array
        if np.isscalar(t):
            tt=[]
            tt.append(t)
        else:
            tt=t
        # main loop through all floating point numbers in reduced precision
        for ti in tt:
            sum=0.0
            # if ti=0 the result is 0 since it definitely covers the whole interval
            if float(ti) < 1.0:
                x=gmpy2.next_above(gmpy2.inf(-1))
                y=gmpy2.next_above(x)
                z=gmpy2.next_above(y)
                err=float(ti)*eps
                # Deal with the very first interval [x,(x+y)/2]
                ctx.precision=53
                ctx.emin=-1023
                ctx.emax=1023
                xmin=float(x)
                xmax=(xmin+float(y))/2.0
                xp=xmin/(1.0-err)
                if xmin < xp < xmax:
                    sum+=self.inputdistribution.pdf(xp)*abs(xp)*eps/(1.0-err)
                # Deal with all standard intervals
                while gmpy2.is_finite(z):
                    ctx.precision=53
                    ctx.emin=-1023
                    ctx.emax=1023
                    xmin=xmax
                    xmax=(float(y) + float(z))/2.0
                    xp=float(y)/(1.0-err)
                    if xmin < float(xp) < xmax:
                        sum+=self.inputdistribution.pdf(xp)*abs(xp)*eps/(1.0-err)
                    ctx.precision=self.precision
                    ctx.emin=self.minexp
                    ctx.emax=self.maxexp
                    x=y
                    y=z
                    z=gmpy2.next_above(z)
                # Deal with the very last interval [x,(x+y)/2]
                xmin=xmax
                xp=float(y)/(1.0-err)
                xmax=float(y)
                if xmin < xp < xmax:
                    sum+=self.inputdistribution.pdf(xp)*abs(xp)*eps/(1.0-err)
            #print('Evaluated at '+repr(ti)+'    Result='+repr(sum))
            sums.append(sum)
        if np.isscalar(t):
            return sum
        else:
            return sums
