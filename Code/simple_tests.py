from pacal import UniformDistr
import gmpy2
from gmpy2 import mpfr
import matplotlib.pyplot as plt

class TestUniformVariable:

    def __init__(self, lowerbound, upperbound, threshold, precision):
        self.lowerbound=lowerbound
        self.upperbound=upperbound
        self.threshold=threshold
        self.precision=precision
        self.error_prob=0

    def compute(self):
        gmpy2.get_context().precision=self.precision
        x=mpfr(self.threshold)
        y=gmpy2.next_above(x)
        gmpy2.get_context().precision=53
        X=UniformDistr(self.lowerbound,self.upperbound)
        self.error_prob=(X.get_piecewise_pdf().integrate(x,y))/(X.get_piecewise_pdf().integrate(x,self.upperbound))

    def plot_against_precision(self,minprecision,maxprecision):
        prec=[]
        err=[]
        for i in range(minprecision,maxprecision):
            prec.append(i)
            self.precision=i
            self.compute()
            err.append(100*self.error_prob)
        plt.plot(prec,err)
        plt.xlabel('precision (bits)')
        plt.ylabel('probability of error (%)')
        plt.show()

    def plot_against_threshold(self):
        th=[]
        err=[]
        for i in range(1,800):
            th.append(i*(self.upperbound-self.lowerbound)/1000)
            self.threshold=th[i-1]
            self.compute()
            err.append(100*self.error_prob)
        plt.plot(th,err)
        plt.xlabel('threshold')
        plt.ylabel('probability of error (%)')
        plt.show()
