from abstract_error_model import AbstractErrorModel
from project_utils import setCurrentContextPrecision, resetContextDefault, printMPFRExactly
from wrapper_error_model import ErrorModelWrapper

import matplotlib.pyplot as plt
from time import time
import gmpy2
from gmpy2 import mpfr
from pacal import UniformDistr, NormalDistr, BetaDistr
import numpy as np

###
# Exact Error Distribution for Low-Precisions (half-precision and below).
###
class LowPrecisionErrorModel(AbstractErrorModel):

    def __init__(self, input_distribution, precision, exponent, polynomial_precision):
        """
        Constructor interpolates the density function using Chebyshev interpolation
        then uses this interpolation to build a PaCal object:
        the self.distribution attribute which contains all the methods we could possibly want
        Inputs:
        input_distribution: a PaCal object representing the distribution for which we want to compute
                            the rounding error distribution
        precision, exponent: specify the gmpy2 precision environment
        polynomial_precision: the number of exact evaluations of the density function used to
                        build the interpolating polynomial representing it
        """

        super(LowPrecisionErrorModel, self).__init__(input_distribution, precision, exponent, polynomial_precision)
        self.name = "LPError(" + input_distribution.getName() + ")"
        setCurrentContextPrecision(self.precision, self.exponent)
        self.inf_val = mpfr(str(self.input_distribution.range_()[0]))
        self.sup_val = mpfr(str(self.input_distribution.range_()[1]))
        self.max_exp = 2 ** (exponent - 1)
        # TODO: deal with infinities
        self.exp_inf_val = np.floor(np.log(abs(self.inf_val), 2))
        self.exp_sup_val = np.floor(np.log(abs(self.sup_val), 2))
        if not gmpy2.is_finite(self.inf_val):
            self.inf_val = gmpy2.next_above( self.inf_val)
        if not gmpy2.is_finite(self.sup_val):
            self.sup_val = gmpy2.next_below(self.sup_val)
        resetContextDefault()

    def _left_segment(self, x):
        return self._right_segment(x)

    def _middle_segment(self, x):
        sum = 0.0
        err = x * self.unit_roundoff
        setCurrentContextPrecision(self.precision, self.exponent)
        z = mpfr(printMPFRExactly(self.inf_val))
        # Loop through all floating point numbers in reduced precision
        while z <= self.sup_val:
            xp = float(printMPFRExactly(z)) / (1.0 - err)
            sum += self.input_distribution.get_piecewise_pdf()(xp) * abs(xp) * self.unit_roundoff / (1.0 - err)
            z = gmpy2.next_above(z)
        resetContextDefault()
        return sum

    def _right_segment(self, x):
        sum = 0.0
        err = x * self.unit_roundoff
        # self.precision - 1 is the usual (i.e. not gmpy2) precision
        if x >= 0:
            max_mantissa = max(0, np.floor(2 ** (self.precision - 1) * (1 / x - 1) - 0.5))
        else:
            max_mantissa = max(0, np.floor(2 ** (self.precision - 1) * (-1 / x - 1) + 0.5))
        # If max_mantissa = 0 we should not enter any of the mantissa loops
        # Else we will enter a loop and we need to add one because range(i,j) stops at j-1
        if max_mantissa > 0:
            max_mantissa += 1
        # Loop through all floating point numbers in reduced precision such that:
        # 1) they lie in the range of the input distribution
        # 2) mantissa satisfies 1+k/2^p <= 1/t - u
        # Case 1: only negative representable numbers
        if self.sup_val < 0:
            # Loop through exponents
            for i in range(self.exp_inf_val, self.exp_sup_val - 1, -1):
                # Loop through mantissas
                for j in range(0,  max_mantissa):
                    m = -(1 + j / (2 ** self.precision))
                    z = ((2 ** i) * m) / (1.0 - err)
                    sum += self.input_distribution.get_piecewise_pdf()(z) * -z * self.unit_roundoff / (1.0 - err)
        # Case 2: negative and positive representable numbers
        elif self.inf_val < 0:
            # Loop through exponents to 0
            for i in range(self.exp_inf_val, -self.max_exp - 1, -1):
                # Loop through mantissas
                for j in range(0, max_mantissa):
                    m = -(1 + j / (2 ** self.precision))
                    z = ((2 ** i) * m) / (1.0 - err)
                    sum += self.input_distribution.get_piecewise_pdf()(z) * -z * self.unit_roundoff / (1.0 - err)
            # Loop through exponents from 0
            for i in range(-self.max_exp, self.exp_sup_val + 1):
                # Loop through mantissas
                for j in range(0, max_mantissa):
                    m = 1 + j / (2 ** self.precision)
                    z = ((2 ** i) * m) / (1.0 - err)
                    sum += self.input_distribution.get_piecewise_pdf()(z) * z * self.unit_roundoff / (1.0 - err)
        # Case 2: only positive representable numbers
        else:
            # Loop through exponents
            for i in range(self.exp_inf_val, self.exp_sup_val + 1):
                # Loop through mantissas
                for j in range(0, max_mantissa):
                    m = 1 + j / (2 ** self.precision)
                    z = ((2 ** i) * m) / (1.0 - err)
                    sum += self.input_distribution.get_piecewise_pdf()(z) * z * self.unit_roundoff / (1.0 - err)
        return sum

    # infVal is finite value
    def getInitialMinValue(self, infVal):
        if not gmpy2.is_finite(infVal):
            print("Error cannot compute intervals with infinity")
            exit(-1)
        bkpCtx = gmpy2.get_context().copy()

        i = 0
        while not gmpy2.is_finite(gmpy2.next_below(infVal)):
            setCurrentContextPrecision(self.precision + i, self.exponent + i)
            i = i + 1

        prec = printMPFRExactly(gmpy2.next_below(infVal))
        gmpy2.set_context(bkpCtx)
        return prec

    # infVal is finite value
    def getFinalMaxValue(self, supVal):
        if not gmpy2.is_finite(supVal):
            print("Error cannot compute intervals with infinity")
            exit(-1)
        bkpCtx = gmpy2.get_context().copy()

        i = 0
        while not gmpy2.is_finite(gmpy2.next_above(supVal)):
            setCurrentContextPrecision(self.precision + i, self.exponent + i)
            i = i + 1

        prec = printMPFRExactly(gmpy2.next_above(supVal))
        gmpy2.set_context(bkpCtx)
        return prec

def test_LP_error_model():
    exponent = 3
    mantissa = 4
    poly_precision = 200
    t = time()
    D = UniformDistr(2, 8)
    E = LowPrecisionErrorModel(D, mantissa, exponent, poly_precision)
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    t = time()
    D = NormalDistr()
    E = LowPrecisionErrorModel(D, mantissa, exponent, poly_precision)
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    t = time()
    D = BetaDistr(2, 2)
    E = LowPrecisionErrorModel(D, mantissa, exponent, poly_precision)
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    exponent = 5
    mantissa = 11
    poly_precision = 50
    t = time()
    D = UniformDistr(2, 4)
    E = LowPrecisionErrorModel(D, mantissa, exponent, poly_precision)
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    t = time()
    D = NormalDistr()
    E = LowPrecisionErrorModel(D, mantissa, exponent, poly_precision)
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    t = time()
    D = BetaDistr(2, 2)
    E = LowPrecisionErrorModel(D, mantissa, exponent, poly_precision)
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    # test wrapper
    wrapper = ErrorModelWrapper(E)
    print(wrapper.getName())
    s = wrapper.getSampleSet()
    plt.hist(s, range=[-1, 1], density=True)