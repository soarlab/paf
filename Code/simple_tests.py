from pacal import UniformDistr
import gmpy2
from gmpy2 import mpfr
import matplotlib.pyplot as plt

# A class for simple test programs
class Test:
    def __init__(self, threshold, precision):
        self.threshold=threshold
        self.precision=precision
        self.error_prob=0

    def compute(self):
        raise NotImplementedError("Method not implemented, please implement derived class")

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
        plt.savefig("pics/no_op_precision")
        plt.clf()

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
        plt.savefig("pics/no_op_threshold")
        plt.clf()

# A class for testing a variable against a threshold
class TestUniformVariable(Test):

    def __init__(self, lowerbound, upperbound, threshold, precision):
        self.lowerbound=lowerbound
        self.upperbound=upperbound
        self.threshold=threshold
        self.precision=precision
        self.error_prob=0

    def compute(self):
        gmpy2.get_context().precision=self.precision
        x=mpfr(self.threshold)
        ceilx=gmpy2.next_above(x)
        gmpy2.get_context().precision=53
        ceilx=(x+ceilx)/2
        X=UniformDistr(self.lowerbound,self.upperbound)
        self.error_prob=(X.get_piecewise_pdf().integrate(x,ceilx))/(X.get_piecewise_pdf().integrate(x,self.upperbound))

# A class for testing the sum of two variables against a threshold
class TestSumUniformVariable(Test):

    def __init__(self, lowerbound1, upperbound1, lowerbound2, upperbound2, threshold, precision):
        self.lowerbound1=lowerbound1
        self.upperbound1=upperbound1
        self.lowerbound2=lowerbound2
        self.upperbound2=upperbound2
        self.threshold=threshold
        self.precision=precision
        self.error_prob=0

    def compute(self):
        gmpy2.get_context().precision=self.precision
        x=mpfr(self.threshold)
        ceilx=gmpy2.next_above(x)
        floorx=gmpy2.next_below(x)
        gmpy2.get_context().precision=53
        ceilx=(x+ceilx)/2
        floorx=(floorx+x)/2
        X=UniformDistr(self.lowerbound1,self.upperbound1)
        Y=UniformDistr(self.lowerbound2,self.upperbound2)
        Z=X+Y
        self.error_prob=(((ceilx-x)/(ceilx-floorx))*Z.get_piecewise_pdf().integrate(x,ceilx))
