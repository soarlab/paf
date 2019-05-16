#import pacal
import gmpy2
from gmpy2 import mpfr

# A class for quantized distributions
class QDIst(object):

    def __init__(self, precision, exponent, distribution):
        self.precision=precision
        self.exponent=exponent
        self.pdfx = []
        gmpy2.get_context().precision=self.precision
        gmpy2.get_context().emax=pow(2,self.exponent-1)
        gmpy2.get_context().emin=-pow(2,self.exponent-1)+1
        print(gmpy2.get_context())
        n=gmpy2.inf(-1)
        self.pdfx.append(n)
        print(n)
        n=gmpy2.next_above(n)
        self.pdfx.append(n)
        print(n)
        while gmpy2.is_finite(n):
            n=gmpy2.next_above(n)
            self.pdfx.append(n)
        n=gmpy2.next_below(n)
        print(n)
        # self.pdfy = []
        # gmpy2.get_context().precision=precision
