import pacal
import matplotlib.pyplot as plt
import numpy as np
from pychebfun import Chebfun
import gmpy2
from gmpy2 import mpfr

class ErrorModel:

    def __init__(self, inputdistribution, precision, minexp, maxexp):
        self.inputdistribution = inputdistribution
        self.precision=precision
        self.minexp=minexp
        self.maxexp=maxexp
        self.cdf=Chebfun.from_function(lambda t:self.__getcdf(t))


    def __getcdf(self,t):
        ctx=gmpy2.get_context()
        ctx.precision=self.precision
        ctx.emin=self.minexp
        ctx.emax=self.maxexp
        x=gmpy2.next_above(gmpy2.inf(-1))
        while gmpy2.is_finite(x):
            ctx.precision=53
            ctx.emin=-1073741823
            ctx.emax=1073741823
            if x<0:
                sum+=self.inputdistribution.get_piecewise_pdf().integrate(float(x)/(1-t),float(x)/(1+t))
            elif x=0:
                sum+=0
            else:
                sum+=self.inputdistribution.get_piecewise_pdf().integrate(float(x)/(1+t),float(x)/(1-t))
            ctx.precision=self.precision
            ctx.emin=self.minexp
            ctx.emax=self.maxexp
            x=gmpy2.next_above(x)
        return sum
