from .abstract_error_model import AbstractErrorModel

from project_utils import set_context_precision, reset_default_precision, printMPFRExactly

import gmpy2
from gmpy2 import mpfr
import numpy as np
from math import floor, log

###
# Exact Error Distribution for Low-Precisions (half-precision and below).
###
class LowPrecisionErrorModel(AbstractErrorModel):

    def __init__(self, input_distribution, input_name, precision, exponent, polynomial_precision=[0, 0]):
        """
        Constructor interpolates the density function using Chebyshev interpolation
        then uses this interpolation to build a PaCal object:
        the self.distribution attribute which contains all the methods we could possibly want
        Inputs:
        input_distribution: a PaCal object representing the distribution for which we want to compute
                            the rounding error distribution
        precision, exponent: specify the gmpy2 precision environment
        polynomial_precision: default precision as implemented in AbstractErrorModel will typically not converge
                              so it is re-implemented as dynamically setting polynomial_precision. For very low
                              precision, polynomial_precision needs to be high because the function is very
                              discontinuous (there are few floating points so it won't impact performance).
                              For higher precision it needs to be low for performance reason (but it won't impact
                              accuracy because the function becomes much more regular).
        Warning: the relative error is not defined in the interval rounding to 0. In low precision this interval might
                 have a large probability. This will be reflected by the distribution not integrating to 1.
                 Example: Uniform(-2,2) with 3 bit exponent, 4 bit mantissa and default polynomial_precision integrates
                 to 0.926 !
        """

        super(LowPrecisionErrorModel, self).__init__(input_distribution, precision, exponent, polynomial_precision)
        #self.name = "LPError(" + input_distribution.getName() + ")"
        self.name = "LPE_" + input_name
        set_context_precision(self.precision, self.exponent)
        self.inf_val = mpfr(str(self.input_distribution.range_()[0]))
        self.sup_val = mpfr(str(self.input_distribution.range_()[1]))
        if not gmpy2.is_finite(self.inf_val):
            self.inf_val = gmpy2.next_above(self.inf_val)
        if not gmpy2.is_finite(self.sup_val):
            self.sup_val = gmpy2.next_below(self.sup_val)
        self.max_exp = 2 ** (exponent - 1)
        if self.inf_val == 0:
            # take most negative exponent
            self.exp_inf_val = -self.max_exp
        else:
            self.exp_inf_val = floor(log(abs(self.inf_val), 2))
        if self.sup_val == 0:
            # take most negative exponent
            self.exp_sup_val = -self.max_exp
        else:
            self.exp_sup_val = floor(log(abs(self.sup_val), 2))
        reset_default_precision()
        if polynomial_precision == [0, 0]:
            self.polynomial_precision = [floor(400.0 / float(self.precision)), floor(100.0 / float(self.precision))]

    def _left_segment(self, x):
        return self._right_segment(x)

    def _middle_segment(self, x):
        sum = 0.0
        err = x * self.unit_roundoff
        set_context_precision(self.precision, self.exponent)
        z = mpfr(printMPFRExactly(self.inf_val))
        # Loop through all floating point numbers in reduced precision
        while z <= self.sup_val:
            xp = float(printMPFRExactly(z)) / (1.0 - err)
            sum += self.input_distribution.get_piecewise_pdf()(xp) * abs(xp) * self.unit_roundoff / (1.0 - err)
            z = gmpy2.next_above(z)
        reset_default_precision()
        return sum

    def _right_segment(self, x):
        sum = 0.0
        err = x * self.unit_roundoff
        min_mantissa = 0
        # self.precision - 1 is the usual (i.e. not gmpy2) precision
        if x >= 0:
            # add one because range(i,j) stops at j-1
            max_mantissa = floor(2 ** (self.precision - 1) * (1 / x - 1) - 0.5) + 1
        else:
            max_mantissa = min(floor(2 ** (self.precision - 1) * (-1 / x - 1) + 0.5) + 1, 2 ** (self.precision-1))
            # for negative values, the test to allow the 0 mantissa is different
            if abs(x) <= (2**self.precision) / (2**(self.precision+1)-1):
                min_mantissa = 0
            else:
                min_mantissa = 1

        # Loop through all floating point numbers in reduced precision such that:
        # 1) they lie in the range of the input distribution
        # 2) mantissa k satisfies 1+k/2^p <= 1/x - u
        # Case 1: only negative representable numbers
        if self.sup_val < 0:
            # Loop through exponents
            for i in range(self.exp_inf_val, self.exp_sup_val - 1, -1):
                # Loop through mantissas
                for j in range(min_mantissa,  max_mantissa):
                    m = -(1 + j / (2 ** (self.precision - 1)))
                    z = ((2 ** i) * m) / (1.0 - err)
                    sum += self.input_distribution.get_piecewise_pdf()(z) * -z * self.unit_roundoff / (1.0 - err)
        # Case 2: negative and positive representable numbers
        elif self.inf_val < 0:
            # Loop through exponents to 0
            for i in range(self.exp_inf_val, -self.max_exp, -1):
                # Loop through mantissas
                for j in range(min_mantissa, max_mantissa):
                    m = -(1 + j / (2 ** (self.precision - 1)))
                    z = ((2 ** i) * m) / (1.0 - err)
                    sum += self.input_distribution.get_piecewise_pdf()(z) * -z * self.unit_roundoff / (1.0 - err)
            # Loop through exponents from 0
            for i in range(1-self.max_exp, self.exp_sup_val + 1):
                # Loop through mantissas
                for j in range(min_mantissa, max_mantissa):
                    m = 1 + j / (2 ** (self.precision - 1))
                    z = ((2 ** i) * m) / (1.0 - err)
                    sum += self.input_distribution.get_piecewise_pdf()(z) * z * self.unit_roundoff / (1.0 - err)
        # Case 2: only positive representable numbers
        else:
            # Loop through exponents
            for i in range(self.exp_inf_val, self.exp_sup_val + 1):
                # Loop through allowed mantissas
                for j in range(min_mantissa, max_mantissa):
                    m = 1 + j / (2 ** (self.precision-1))
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
            set_context_precision(self.precision + i, self.exponent + i)
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
            set_context_precision(self.precision + i, self.exponent + i)
            i = i + 1

        prec = printMPFRExactly(gmpy2.next_above(supVal))
        gmpy2.set_context(bkpCtx)
        return prec
