import pacal
import matplotlib.pyplot as plt
import numpy as np
from pychebfun import Chebfun
import gmpy2
from gmpy2 import mpfr
import random

class ErrorModel:

    def __init__(self, inputdistribution, precision, minexp, maxexp):
        self.inputdistribution = inputdistribution
        self.precision=precision
        self.minexp=minexp
        self.maxexp=maxexp
        #for i in range(1,20):
        #    x=self.__getpdf(-1.0+i*0.1)
        #    print(x)
        #pdf=Chebfun.from_function(lambda t:self.__getpdf(t),N=10)
        #print(repr(pdf))
        #print(self.__getcdf(1))
        print(self.__getcdf(-1))
        #for i in range(1,20):
        #    x=self.__getcdf(-1.0+i*0.1)
        #    print('f('+repr(-1.0+i*0.1)+')='+repr(x))
        #cdf=Chebfun.from_function(lambda t:self.__getcdf(t),N=100)


    def __getpdf(self, t):
        #if t<-1 or t>1:
        #    raise Exception('|t| should not exceed 1. The value of x was: {}'.format(t))
        ctx=gmpy2.get_context()
        ctx.precision=self.precision
        ctx.emin=self.minexp
        ctx.emax=self.maxexp
        eps=2**-self.precision
        x=gmpy2.next_above(gmpy2.inf(-1))
        sum=0.0
        pdf=self.inputdistribution.get_piecewise_pdf()
        while gmpy2.is_finite(x):
            ctx.precision=53
            ctx.emin=-1023
            ctx.emax=1023
            #z=random.random()
            if x<0:
                y=float(x)
                sum=sum+pdf(y/(1+t*eps))*(-y*eps/(1-t*eps)**2)
                #if z<0.03:
                #    print('at: ' + repr(x) +'       fct:'+repr(pdf(-y/(1+t*eps))*(y*eps/(1+t*eps)**2))+'        sum: '+repr(sum))
            elif x>0:
                y=float(x)
                sum=sum+pdf(y/(1-t*eps))*(y*eps/(1-t*eps)**2)
                #if z<0.03:
                #    print('at: ' + repr(x) +'       fct:'+repr(pdf(y/(1-t*eps))*(y*eps/(1-t*eps)**2))+'        sum: '+repr(sum))
            ctx.precision=self.precision
            ctx.emin=self.minexp
            ctx.emax=self.maxexp
            x=gmpy2.next_above(x)
        return(sum)
        #print(sum)


    def __getcdf(self,t):
        ctx=gmpy2.get_context()
        ctx.precision=self.precision
        ctx.emin=self.minexp
        ctx.emax=self.maxexp
        eps=2**-(self.precision)
        x=gmpy2.next_above(gmpy2.inf(-1))
        y=gmpy2.next_above(x)
        z=gmpy2.next_above(y)
        sum=0.0
        while gmpy2.is_finite(y):
            ctx.precision=53
            ctx.emin=-1023
            ctx.emax=1023
            xx=float(x)
            yy=float(y)
            zz=float(z)
            d=yy-(yy-xx)/2
            if yy<0:
                u=max(d,min(yy/(1+t*eps), yy+(zz-yy)/2))
                tmp=self.inputdistribution.cdf(u)-self.inputdistribution.cdf(d)
                sum+=tmp
                if tmp!=0:
                    print('at: '+repr(d)+' < ' + repr(yy)+' < ' +repr(u)+'       fct:'+repr(tmp)+'        sum: '+repr(sum))
            elif yy>0:
                u=max(d,min(yy/(1-t*eps), yy+(zz-yy)/2))
                tmp=self.inputdistribution.cdf(u)-self.inputdistribution.cdf(d)
                sum+=tmp
                if tmp!=0:
                    print('xx='+repr(xx)+', yy='+repr(yy)+', zz='+repr(zz)+', rel err='+repr(yy/(1-t*eps)))
                    print('at: '+repr(d)+' < ' + repr(yy)+' < ' +repr(u)+'   '+repr(yy+(zz-yy)/2)+'       fct:'+repr(tmp)+'        sum: '+repr(sum))
            ctx.precision=self.precision
            ctx.emin=self.minexp
            ctx.emax=self.maxexp
            x=gmpy2.next_above(x)
            y=gmpy2.next_above(x)
            z=gmpy2.next_above(y)
        return sum
