import pacal
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as cheb
import numpy as np
from pychebfun import Chebfun
import gmpy2
from gmpy2 import mpfr
import random
import os

class ErrorModel:

    def __init__(self, inputdistribution, precision, minexp, maxexp):
        self.inputdistribution = inputdistribution
        self.precision=precision
        self.minexp=minexp
        self.maxexp=maxexp


    def compute(self):
        #self.cdf=Chebfun.from_function(lambda t:self.__getcdf(t), N=42)
        self.pdf=cheb.Chebyshev.interpolate(lambda t:self.__getpdf(t),deg=42)
        #self.pdf=Chebfun.from_function(lambda t:self.__getpdf(t), N=64  )
        x=np.linspace(-1,1,21)
        z=self.pdf(x)
        print(repr(self.pdf.integ(lbnd=-1)(1)))
        plt.plot(x, z)
        #print(repr(self.pdf.sum()))
        strFile ='pics/TH_'+repr(self.inputdistribution.getName()).replace("'",'')+'_'+repr(self.precision)
        plt.savefig(strFile)
        plt.clf()


    def __getpdf(self, t):
        ctx=gmpy2.get_context()
        ctx.precision=self.precision
        ctx.emin=self.minexp
        ctx.emax=self.maxexp
        eps=2**-self.precision
        sums=[]
        #test if  the input is scalar of an array
        if np.isscalar(t):
            tt=[]
            tt.append(t)
        else:
            tt=t
        for ti in tt:
            sum=0.0
            x=gmpy2.next_above(gmpy2.inf(-1))
            y=gmpy2.next_above(x)
            z=gmpy2.next_above(y)
            while gmpy2.is_finite(x):
                ctx.precision=53
                ctx.emin=-1023
                ctx.emax=1023
                xx=float(x)
                yy=float(y)
                zz=float(z)
                d=yy-(yy-xx)/2
                if yy<0:
                    if yy/(1+ti*eps)>d:
                        if yy/(1+ti*eps)<yy+(zz-yy)/2:
                            sum+=self.inputdistribution.pdf(yy/(1+ti*eps))*(-yy*eps/(1+ti*eps)**2)
                elif yy>0:
                    if yy/(1-ti*eps)>d:
                        if yy/(1-ti*eps)<yy+(zz-yy)/2:
                            sum+=self.inputdistribution.pdf(yy/(1-ti*eps))*(yy*eps/(1-ti*eps)**2)
                ctx.precision=self.precision
                ctx.emin=self.minexp
                ctx.emax=self.maxexp
                x=gmpy2.next_above(x)
                y=gmpy2.next_above(x)
                z=gmpy2.next_above(y)
            print('Evaluated at '+repr(ti)+'    Result='+repr(sum))
            sums.append(sum)
        if np.isscalar(t):
            return sum
        else:
            return sums


    def __getcdf(self,t):
        ctx=gmpy2.get_context()
        ctx.precision=self.precision
        ctx.emin=self.minexp
        ctx.emax=self.maxexp
        eps=2**-(self.precision)
        sums=[]
        #test if  the input is scalar of an array
        if np.isscalar(t):
            tt=[]
            tt.append(t)
        else:
            tt=t
        for ti in tt:
            x=gmpy2.next_above(gmpy2.inf(-1))
            y=gmpy2.next_above(x)
            z=gmpy2.next_above(y)
            sum = 0.0

            while gmpy2.is_finite(y):
                ctx.precision=53
                ctx.emin=-1023
                ctx.emax=1023
                xx=float(x)
                yy=float(y)
                zz=float(z)
                d=yy-(yy-xx)/2
                if yy<0:
                    if yy/(1+ti*eps)>d:
                        if yy/(1+ti*eps)<yy+(zz-yy)/2:
                            u=yy/(1+ti*eps)
                        else:
                            u=yy+(zz-yy)/2
                        sum+=self.inputdistribution.cdf(u)-self.inputdistribution.cdf(d)
                elif yy>0:
                    if yy/(1-ti*eps)>d:
                        if yy/(1-ti*eps)<yy+(zz-yy)/2:
                            u=yy/(1-ti*eps)
                        else:
                            u=yy+(zz-yy)/2
                            tmp=self.inputdistribution.cdf(u)-self.inputdistribution.cdf(d)
                            sum+=tmp
                ctx.precision=self.precision
                ctx.emin=self.minexp
                ctx.emax=self.maxexp
                x=gmpy2.next_above(x)
                y=gmpy2.next_above(x)
                z=gmpy2.next_above(y)

            print('Evaluated at '+repr(ti)+'    Result='+repr(sum))
            sums.append(sum)
        if np.isscalar(t):
            return sum
        else:
            return sums
