import math
from abc import ABC
from time import time
import matplotlib.pyplot as plt
import gmpy2
from gmpy2 import mpfr
from numpy import isscalar, zeros_like, asfarray
from pacal import ConstDistr, NormalDistr, UniformDistr, BetaDistr
from pacal.distr import Distr
from pacal.segments import PiecewiseDistribution, Segment
from pacal.utils import wrap_pdf
from pychebfun import chebfun

import model
import matplotlib
from scipy import integrate
from scipy.stats import kstest
from project_utils import printMPFRExactly
import numpy as np


###
# Functions switching between low and double precision. Beware mantissa = gmpy2 precision (includes sign bit)
###

def set_context_precision(mantissa, exponent):
    ctx = gmpy2.get_context()
    if mantissa is None:
        ctx.precision = 24
    else:
        ctx.precision = mantissa
    if exponent is None:
        ctx.emax = 2 ** 7
    else:
        ctx.emax = 2 ** (exponent - 1)
    ctx.emin = 1 - ctx.emax


def reset_default_precision():
    gmpy2.set_context(gmpy2.context())


###
# Abstract ErrorModel class.
###

class ErrorModel(Distr):
    def __init__(self, input_distribution, precision, exponent, polynomial_precision=None):
        """
        Error distribution class.
        Inputs:
            input_distribution: a PaCal object representing the distribution for which we want to compute
                                the rounding error distribution
            precision, exponent: gmpy2 precision environment
            polynomial_precision: a 3-tuple of integer controlling the precision of the polynomial interpolation of
                                  each segment. Default is None (no error model with no interpolation).
        """
        super(ErrorModel, self).__init__()
        self.input_distribution = input_distribution
        if input_distribution is None:
            self.name = "ErrorModel"
        else:
            self.name = "Error(" + input_distribution.getName() + ")"
            if self.input_distribution.piecewise_pdf is None:
                self.input_distribution.init_piecewise_pdf()
        # gmpy2 precision (or None):
        self.precision = precision
        self.polynomial_precision = polynomial_precision
        self.exponent = exponent
        # In gmpy2 precision includes a sign bit, so 2 ** precision = unit roundoff
        if self.precision is None:
            self.unit_roundoff = 2 ** (-24)
        else:
            self.unit_roundoff = 2 ** (-self.precision)

    def init_piecewise_pdf(self):
        """Initialize the pdf represented as a piecewise function.
        This method should be overridden by subclasses."""
        raise NotImplementedError()

    def _left_segment(self, x):
        """
        Abstract method for the [-1,-0.5] segment of the pdf
        Input: t a real between -1 and -0.5
        """
        raise NotImplementedError()

    def _middle_segment(self, x):
        """
        Abstract method for the [-0.5,1] segment of the pdf
        Input: t a real between -0.5 and 0.5
        """
        raise NotImplementedError()

    def _right_segment(self, x):
        """
        Abstract method for the [-0.5,1] segment of the pdf
        Input: t a real between -0.5 and 0.5
        """
        raise NotImplementedError()

    def init_piecewise_pdf(self):
        piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        piecewise_pdf.addSegment(Segment(-1, -0.5, wrapped_pdf))
        piecewise_pdf.addSegment(Segment(-0.5, 0.5, wrapped_pdf))
        piecewise_pdf.addSegment(Segment(0.5, 1, wrapped_pdf))
        self.piecewise_pdf = piecewise_pdf.toInterpolated()

    def pdf(self, x):
        if isscalar(x):
            if -1 <= x < -0.5:
                return self._left_segment(x)
            elif -0.5 <= x <= 0.5:
                return self._middle_segment(x)
            elif 0.5 < x <= 1:
                return self._right_segment(x)
            else:
                return 0.0
        else:
            y = zeros_like(asfarray(x))
            for index, ti in enumerate(x):
                if -1 <= ti < -0.5:
                    y[index] = self._left_segment(ti)
                elif -0.5 <= ti <= 0.5:
                    y[index] = self._middle_segment(ti)
                elif 0.5 < ti <= 1:
                    y[index] = self._right_segment(ti)
            return y

    def rand_raw(self, n=None):  # None means return scalar
        inv_cdf = self.get_piecewise_invcdf()
        u = np.random.uniform(size=n)
        return inv_cdf(u)

    def __call__(self, x):
        return self.pdf(self, x)

    def range(self):
        return -1.0, 1.0

    def getName(self):
        return self.name

    def __str__(self):
        if self.p is None:
            return "Error#{0}".format(self.id())
        else:
            return "Error(p={0})#{1}".format(self.p, self.id())

    def compare(self, n=10000, file_name=None):
        """
        A function to compare the ErrorModel density function with an empirical distribution of relative errors
        and return a K-S test
        :param n: number of samples
        :param file_name: optional, if not None the graph will be saved using the file_name + name of the distribution.
        :return: the Kolmogorov-Smirnov (K-S) statistic and p-value
        """
        if self.input_distribution is None:
            return "Nothing to compare against!"
        empirical = self.input_distribution.rand(n)
        pdf = self.get_piecewise_pdf()
        cdf = self.get_piecewise_cdf()
        rounded = np.zeros_like(empirical)
        set_context_precision(self.precision, self.exponent)
        for index, ti in enumerate(empirical):
            rounded[index] = mpfr(str(empirical[index]))
        reset_default_precision()
        empirical = (empirical - rounded) / (empirical * self.unit_roundoff)
        ks_test = kstest(empirical, cdf)
        x = np.linspace(-1, 1, 201)
        matplotlib.pyplot.close("all")
        matplotlib.rcParams.update({'font.size': 12})
        plt.hist(empirical, bins=2 * math.floor(n ** (1 / 3)), range=[-1, 1], density=True)
        y = pdf(x)
        h = plt.plot(x, y)
        plt.title(
            self.input_distribution.getName() + ", KS-test=" + str(round(ks_test[0], 4)) + ", p-val=" + str(round(ks_test[1], 4)))
        if file_name is None:
            plt.show()
        else:
            plt.savefig("file_name" + self.getName() + ".png")
        return ks_test


###
# Exact Error Distribution for High-Precisions (above half-precision).
###

class HighPrecisionErrorModel(ErrorModel):

    def __init__(self, input_distribution, precision, exponent):
        """
        The class implements the high-precision error distribution function.
        Inputs:
            input_distribution: a PaCal object representing the distribution for which we want to compute
                            the rounding error distribution
            precision, exponent: gmpy2 precision environment
        """
        super(HighPrecisionErrorModel, self).__init__(input_distribution, precision, exponent)
        self.name = "HPError(" + input_distribution.getName() + ")"
        self.central_constant = None
        self._get_min_exponent()
        self._get_max_exponent()
        self._compute_central_constant()

    def _left_segment(self, x):
        """
        :param x: SCALAR real such that -1.0 < x <= -0.5. Arrays are dealt with in self.pdf.
        :return: pdf evaluated at (x)
        """
        return self._compute_integral(x) / ((1 - self.unit_roundoff * x) ** 2)

    def _middle_segment(self, x):
        """
        :param x: SCALAR real such that abs(x) <= 0.5. Arrays are dealt with in self.pdf.
        :return: pdf evaluated at (x)
        """
        return self.central_constant / ((1 - self.unit_roundoff * x) ** 2)

    def _right_segment(self, x):
        """
        :param x: SCALAR real such that 0.5 < x <= 1. Arrays are dealt with in self.pdf.
        :return: pdf evaluated at (x)
        """
        return self._compute_integral(x) / ((1 - self.unit_roundoff * x) ** 2)

    def _compute_central_constant(self):
        self.central_constant = self._compute_integral()

    def _compute_integral(self, t=None):
        """
        Compute the quantity \sum_e \int_{2^e}^{2^{next(e)}} f(t) t/(\alpha*2^{e}) dt
        where f is the pdf of input_distribution
        """
        emax = 2 ** (self.exponent - 1)
        emin = 1 - emax
        S = 0.0
        I = 0.0
        if t is None:
            alpha = 2
        else:
            alpha = (1 / abs(t) - self.unit_roundoff)
        f = self.input_distribution.get_piecewise_pdf()
        # test if the range of input_distribution covers 0
        if self.min_sign < self.max_sign:
            # sum from -2^self.min_exp to -2^emin
            e = self.min_exp
            while e > emin:
                I = integrate.quad(lambda x: -x * f(x) / 2 ** float(e), -alpha * 2 ** float(e - 1), -2 ** float(e - 1))
                S = S + I[0]
                e -= 1
            # sum from 2^emin to 2^self.emax
            e = emin
            while e < self.max_exp:
                I = integrate.quad(lambda x: x * f(x) / 2 ** float(e + 1), 2 ** float(e), alpha * 2 ** float(e))
                S = S + I[0]
                e += 1
        elif self.max_sign < 0:
            # sum from -2^self.min_exp to -2^self.max_exp
            e = self.min_exp
            while e > self.max_exp:
                I = integrate.quad(lambda x: -x * f(x) / 2 ** float(e), -alpha * 2 ** float(e - 1), -2 ** float(e - 1))
                S = S + I[0]
                e -= 1
        else:
            # sum from 2^self.min_exp to 2^self.max_exp
            e = self.min_exp
            while e < self.max_exp:
                I = integrate.quad(lambda x: x * f(x) / 2 ** float(e + 1), 2 ** float(e), alpha * 2 ** float(e))
                S = S + I[0]
                e += 1
        return S

    def _get_min_exponent(self):
        set_context_precision(self.precision, self.exponent)
        inf_val = mpfr(str(self.input_distribution.range_()[0]))
        self.min_sign = gmpy2.sign(inf_val)
        # For some reason the exponent returned by get_exp() is 1 too high and 0 for infinities
        if gmpy2.is_finite(inf_val):
            e = gmpy2.get_exp(inf_val) - 1
        else:
            e = 2 ** (self.exponent - 1)
        if self.min_sign > 0:
            self.min_exp = e
        else:
            if inf_val < -2 ** (float)(e):
                self.min_exp = e + 1
            else:
                self.min_exp = e
        reset_default_precision()

    def _get_max_exponent(self):
        set_context_precision(self.precision, self.exponent)
        sup_val = mpfr(str(self.input_distribution.range_()[1]))
        self.max_sign = gmpy2.sign(sup_val)
        # For some reason the exponent returned by get_exp() is 1 too high and 0 if sup_val is infinite
        if gmpy2.is_finite(sup_val):
            e = gmpy2.get_exp(sup_val) - 1
        else:
            e = 2 ** (self.exponent - 1)
        if self.max_sign < 0:
            self.max_exp = e
        else:
            if sup_val > 2 ** float(e):
                self.max_exp = e + 1
            else:
                self.max_exp = e
        reset_default_precision()


###
# Exact Error Distribution for Low-Precisions (half-precision and below).
###

class LowPrecisionErrorModel(ErrorModel):

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
        set_context_precision(self.precision, self.exponent)
        self.inf_val = mpfr(str(self.input_distribution.range_()[0]))
        self.sup_val = mpfr(str(self.input_distribution.range_()[1]))
        self.max_exp = 2 ** (exponent - 1)
        # TODO: deal with infinities
        self.exp_inf_val = math.floor(math.log(abs(self.inf_val), 2))
        self.exp_sup_val = math.floor(math.log(abs(self.sup_val), 2))
        if not gmpy2.is_finite(self.inf_val):
            self.inf_val = gmpy2.next_above( self.inf_val)
        if not gmpy2.is_finite(self.sup_val):
            self.sup_val = gmpy2.next_below(self.sup_val)
        reset_default_precision()

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
        # self.precision - 1 is the usual (i.e. not gmpy2) precision
        if x >= 0:
            max_mantissa = max(0, math.floor(2 ** (self.precision - 1) * (1 / x - 1) - 0.5))
        else:
            max_mantissa = max(0, math.floor(2 ** (self.precision - 1) * (-1 / x - 1) + 0.5))
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


###
# Approximate Error Model given by the "Typical Distribution"
###

class TypicalErrorModel(ErrorModel, ABC):
    """
    An implementation of the typical error distribution with three segments
    """
    def __init__(self, input_distribution=None, precision=None, **kwargs):
        super(TypicalErrorModel, self).__init__(input_distribution, precision, None)
        self.name = "TypicalErrorDistribution"
        if precision is not None:
            self.p = precision-1
        else:
            self.p = None

    def _left_segment(self, x):
        if self.p is None:
            y = 0.5 * ((1.0 / x) - 1.0) + 0.25 * (((1.0 / x) - 1.0) ** 2)
        else:
            u = 2 ** (-self.p - 1)
            alpha = np.floor(2 ** self.p * (-1 / x - 1) + 0.5)
            y = 1 / (2 ** self.p * (1 - u * x) ** 2) * (
                    2 / 3 + 0.5 * alpha + 2 ** (-self.p - 2) * alpha * (alpha - 1))
        return y

    def _middle_segment(self, x):
        if self.p is None:
            y = 0.75
        else:
            u = 2 ** (-self.p - 1)
            y = 1 / (2 ** self.p * (1 - u * x) ** 2) * (2 / 3 + 3 * (2 ** self.p - 1) / 4)
        return y

    def _right_segment(self, x):
        if self.p is None:
            y = 0.5 * ((1.0 / x) - 1.0) + 0.25 * (((1.0 / x) - 1.0) ** 2)
        else:
            u = 2 ** (-self.p - 1)
            alpha = np.floor(2 ** self.p * (1 / x - 1) - 0.5)
            y = 1 / (2 ** self.p * (1 - u * x) ** 2) * (
                    2 / 3 + 0.5 * alpha + 2 ** (-self.p - 2) * alpha * (alpha - 1))
        return y

class ErrorModelPointMass:
    def __init__(self, wrapperInputDistribution, precision, exponent):
        self.wrapperInputDistribution = wrapperInputDistribution
        self.inputdistribution = self.wrapperInputDistribution.execute()
        self.inputdistribution.get_piecewise_pdf()
        self.precision = precision
        self.exponent = exponent
        self.unit_roundoff = 2 ** -self.precision
        set_context_precision(self.precision, self.exponent)
        qValue = printMPFRExactly(mpfr(str(self.inputdistribution.rand(1)[0])))
        reset_default_precision()
        error = float(str(self.inputdistribution.rand(1)[0])) - float(qValue)
        self.distribution = ConstDistr(float(error))
        self.distribution.get_piecewise_pdf()

    def execute(self):
        return self.distribution


###
# Wrapper class for all Error Models
###

class ErrorModelWrapper:
    """
    Wrapper class implementing only the methods which are required from tree_model
    Input: an ErrorModel object
    """
    def __init__(self, error_model):
        self.error_model = error_model
        self.sampleInit = True

    def __str__(self):
        return self.error_model.getName()

    def getName(self):
        return self.error_model.getName()

    def execute(self):
        return self.self.error_model

    def getSampleSet(self, n=100000):
        # it remembers values for future operations
        if self.sampleInit:
            self.sampleSet = self.error_model.rand(n)
            self.sampleInit = False
        return self.sampleSet


###
# TESTS
###

def test_HP_error_model():
    exponent = 8
    mantissa = 24
    t = time()
    U = UniformDistr(4, 32)
    E = HighPrecisionErrorModel(U, mantissa, exponent)
    E.init_piecewise_pdf()
    print(E.getName())
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    t = time()
    U = UniformDistr(4, 5)
    E = HighPrecisionErrorModel(U, mantissa, exponent)
    E.init_piecewise_pdf()
    print(E.getName())
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    t = time()
    U = UniformDistr(7, 8)
    E = HighPrecisionErrorModel(U, mantissa, exponent)
    E.init_piecewise_pdf()
    print(E.getName())
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    t = time()
    U = NormalDistr()
    E = HighPrecisionErrorModel(U, mantissa, exponent)
    E.init_piecewise_pdf()
    print(E.getName())
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    # test wrapper
    wrapper = ErrorModelWrapper(E)
    print(wrapper.getName())
    s = wrapper.getSampleSet()
    plt.hist(s, range=[-1, 1], density=True)


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


def test_typical_error_model():
    t = time()
    E = TypicalErrorModel()
    E.init_piecewise_pdf()
    print(E.getName())
    print(E.int_error())
    # Test comparing with nothing. Should return the string N/A
    print(E.compare())
    print(time() - t)
    U = UniformDistr(4, 32)
    E = TypicalErrorModel(U)
    E.init_piecewise_pdf()
    print(E.getName())
    print(E.int_error())
    # Test comparing with U, precision unspecified
    print(E.compare())
    print(time() - t)
    E = TypicalErrorModel(U, 9)
    E.init_piecewise_pdf()
    print(E.getName())
    print(E.int_error())
    # Test comparing with U, precision specified
    print(E.compare())
    print(time() - t)
    # A distribution badly approximated by the TypicalErrorModel
    U = UniformDistr(4, 6)
    E = TypicalErrorModel(U)
    E.init_piecewise_pdf()
    print(E.getName())
    print(E.int_error())
    # Comparison should show poor KS-statistics
    print(E.compare())
    print(time() - t)
    # test wrapper
    wrapper = ErrorModelWrapper(E)
    print(wrapper.getName())
    s = wrapper.getSampleSet()
    plt.hist(s, range=[-1, 1], density=True)



###
# OLD CODE
###

# my_pdf = None
#
#
# def genericPdf(x):
#     if isinstance(x, float) or isinstance(x, int) or len(x) == 1:
#         if x < -1 or x > 1:
#             return 0
#         else:
#             return my_pdf(x)
#     else:
#         res = np.zeros(len(x))
#         for index, ti in enumerate(x):
#             if ti < -1 or ti > 1:
#                 res[index] = 0
#             else:
#                 res[index] = my_pdf(ti)
#         return res
#     exit(-1)
#
# def getTypical(x):
#     if isinstance(x, float) or isinstance(x, int) or len(x) == 1:
#         if abs(x) <= 0.5:
#             return 0.75
#         else:
#             return 0.5 * ((1.0 / x) - 1.0) + 0.25 * (((1.0 / x) - 1.0) ** 2)
#     else:
#         res = np.zeros(len(x))
#         for index, ti in enumerate(x):
#             if abs(ti) <= 0.5:
#                 res[index] = 0.75
#             else:
#                 res[index] = 0.5 * ((1.0 / ti) - 1.0) + 0.25 * (((1.0 / ti) - 1.0) ** 2)
#         return res
#     exit(-1)
#
#
# typVariable = None
#
#
# def createTypical(x):
#     return typVariable(x)


# class WrappedPiecewiseTypicalError():
#
#     def __init__(self, p=None):
#         self.sampleInit = True
#         self.distribution = PiecewiseTypicalError(p)
#         self.distribution.init_piecewise_pdf()
#
#     def execute(self):
#         return self.distribution
#
#     def getSampleSet(self, n=100000):
#         # it remembers values for future operations
#         if self.sampleInit:
#             self.sampleSet = self.distribution.rand(n)
#             self.sampleInit = False
#         return self.sampleSet


# class TypicalErrorModel:
#     def __init__(self, precision, exp, poly_precision):
#         self.poly_precision = poly_precision
#         self.name = "E"
#         self.precision = precision
#         self.exp = exp
#         self.sampleInit = True
#         self.unit_roundoff = 2 ** (-self.precision)
#         self.distribution = self.createTypicalErrorDistr()
#
#     def createTypicalErrorDistr(self):
#         global typVariable
#         typVariable = chebfun(getTypical, domain=[-1.0, 1.0], N=self.poly_precision)
#         self.distribution = FunDistr(createTypical, breakPoints=[-1.0, 1.0], interpolated=True)
#         self.distribution.get_piecewise_pdf()
#         return self.distribution
#
#     def execute(self):
#         return self.distribution
#
#     def getSampleSet(self, n=100000):
#         # it remembers values for future operations
#         if self.sampleInit:
#             self.sampleSet = self.distribution.rand(n)
#             self.sampleInit = False
#         return self.sampleSet
#
#
#
