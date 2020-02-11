import math
import time
import matplotlib.pyplot as plt
import gmpy2
from gmpy2 import mpfr
from numpy import isscalar, zeros_like, asfarray
from pacal import ConstDistr, NormalDistr, UniformDistr, FunDistr
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
# Functions switching between low and double precision
###

def set_context_precision(mantissa, exponent):
    ctx = gmpy2.get_context()
    ctx.precision = mantissa
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
        self.name = "Error(" + input_distribution.name + ")"
        if self.input_distribution.piecewise_pdf is None:
            self.input_distribution.init_piecewise_pdf()
        self.precision = precision
        self.polynomial_precision = polynomial_precision
        self.exponent = exponent
        self.eps = 2 ** (-self.precision)

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
                    return self._left_segment(x)
                elif -0.5 <= ti <= 0.5:
                    return self._middle_segment(x)
                elif 0.5 < ti <= 1:
                    return self._right_segment(x)
            return y

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

    def compare(self, n=100000, file_name=None):
        """
        A function to compare the ErrorModel density function with an empirical distribution of relative errors
        and return a K-S test
        :param n: number of samples
        :param file_name: optional, if not None the graph will be saved using the file_name + name of the distribution.
        :return: the Kolmogorov-Smirnov (K-S) statistic and p-value
        """
        empirical = self.input_distribution.rand(n)
        pdf = self.get_piecewise_pdf()
        cdf = self.get_piecewise_cdf()
        rounded = np.zeros_like(empirical)
        set_context_precision(self.precision, self.exp)
        for index, ti in enumerate(empirical):
            rounded[index] = mpfr(str(empirical[index]))
        reset_default_precision()
        empirical = (empirical - rounded) / (empirical * self.eps)
        ks_test = kstest(empirical, cdf)
        x = np.linspace(-1, 1, 201)
        plt.close()
        matplotlib.rcParams.update({'font.size': 12})
        plt.hist(empirical, bins=2 * math.floor(n ** (1 / 3)), range=[-1, 1], density=True)
        y = pdf(x)
        h = plt.plot(x, y)
        plt.title(
            self.input_distribution.getName() + ", KS-test=" + str(round(KS[0], 4)) + ", p-val=" + str(round(KS[1], 4)))
        if file_name is None:
            plt.show()
        else:
            plt.savefig("file_name" + self.getName() + ".png")
        matplotlib.pyplot.close("all")
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
        super(HighPrecisionErrorModel, input_distribution, precision, exponent, self).__init__()
        self.name = "HPError(" + input_distribution.name + ")"
        self.central_constant = None
        self._get_min_exponent()
        self._get_max_exponent()

    def init_piecewise_pdf(self):
        if self.central_constant is None:
            self._compute_central_constant()
        piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        piecewise_pdf.addSegment(Segment(-1, -0.5, wrapped_pdf))
        piecewise_pdf.addSegment(Segment(-0.5, 0.5, wrapped_pdf))
        piecewise_pdf.addSegment(Segment(0.5, 1, wrapped_pdf))
        self.piecewise_pdf = piecewise_pdf.toInterpolated()

    def rand_raw(self, n=None):  # None means return scalar
        inv_cdf = self.get_piecewise_invcdf()
        u = np.random.uniform(size=n)
        return inv_cdf(u)

    def _left_segment(self, x):
        """
        :param x: SCALAR real such that -1.0 < x <= -0.5. Arrays are dealt with in self.pdf.
        :return: pdf evaluated at (x)
        """
        return self._compute_integral(x) / ((1 - self.eps * x) ** 2)

    def _middle_segment(self, x):
        """
        :param x: SCALAR real such that abs(x) <= 0.5. Arrays are dealt with in self.pdf.
        :return: pdf evaluated at (x)
        """
        return self.central_constant / ((1 - self.eps * x) ** 2)

    def _right_segment(self, x):
        """
        :param x: SCALAR real such that 0.5 < x <= 1. Arrays are dealt with in self.pdf.
        :return: pdf evaluated at (x)
        """
        return self._compute_integral(x) / ((1 - self.eps * x) ** 2)

    def _compute_central_constant(self):
        self.central_constant = self._compute_integral()

    def _compute_integral(self, t=None):
        """
        Compute the quantity \sum_e \int_{2^e}^{2^{next(e)}} f(t) t/(\alpha*2^{e}) dt
        where f is the pdf of input_distribution
        """
        emax = 2 ** (self.exp - 1)
        emin = 1 - emax
        S = 0.0
        I = 0.0
        if t is None:
            alpha = 2
        else:
            alpha = (1 / abs(t) - self.eps)
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
        set_context_precision(self.precision, self.exp)
        inf_val = mpfr(str(self.input_distribution.range_()[0]))
        self.min_sign = gmpy2.sign(inf_val)
        # For some reason the exponent returned by get_exp() is 1 too high and 0 for infinities
        if gmpy2.is_finite(inf_val):
            e = gmpy2.get_exp(inf_val) - 1
        else:
            e = 2 ** (self.exp - 1)
        if self.min_sign > 0:
            self.min_exp = e
        else:
            if inf_val < -2 ** (float)(e):
                self.min_exp = e + 1
            else:
                self.min_exp = e
        reset_default_precision()

    def _get_max_exponent(self):
        set_context_precision(self.precision, self.exp)
        sup_val = mpfr(str(self.input_distribution.range_()[1]))
        self.max_sign = gmpy2.sign(sup_val)
        # For some reason the exponent returned by get_exp() is 1 too high and 0 if sup_val is infinite
        if gmpy2.is_finite(sup_val):
            e = gmpy2.get_exp(sup_val) - 1
        else:
            e = 2 ** (self.exp - 1)
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

        super(LowPrecisionErrorModel, input_distribution, precision, exponent, polynomial_precision, self).__init__()
        self.name = "LPError(" + input_distribution.name + ")"
        set_context_precision(self.precision, self.exponent)
        self.inf_val = mpfr(str(self.input_distribution.range_()[0]))
        self.sup_val = mpfr(str(self.input_distribution.range_()[1]))
        if not gmpy2.is_finite(self.inf_val):
            self.inf_val = gmpy2.next_above( self.inf_val)
        if not gmpy2.is_finite(self.sup_val):
            self.sup_val = gmpy2.next_below(self.sup_val)

    def _left_segment(self, x):
        if x<
            return 0.0

    def _middle_segment(self, x):
        sum = 0.0
        err = x * self.eps
        # loop through all floating point numbers in reduced precision
        x = mpfr(printMPFRExactly(self.inf_val))
        y = gmpy2.next_above(x)
        z = gmpy2.next_above(y)

    def _right_segment(self, x):
        if x>
            return 0.0

    # infVal is finite value
    def getInitialMinValue(self, infVal):
        if not gmpy2.is_finite(infVal):
            print("Error cannot compute intervals with infinity")
            exit(-1)
        bkpCtx = gmpy2.get_context().copy()

        i = 0
        while not gmpy2.is_finite(gmpy2.next_below(infVal)):
            set_context_precision(self.precision + i, self.exp + i)
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
            set_context_precision(self.precision + i, self.exp + i)
            i = i + 1

        prec = printMPFRExactly(gmpy2.next_above(supVal))
        gmpy2.set_context(bkpCtx)
        return prec

    # Compute the exact density
    def getpdf(self, t):
        '''
    Constructs the EXACT probability density function at point t in [-1,1]
    Exact values are used to build the interpolating polynomial
        '''

        set_context_precision(self.precision, self.exp)
        eps = 2 ** -self.precision

        infVal = mpfr(str(self.wrapperInputDistribution.a))
        supVal = mpfr(str(self.wrapperInputDistribution.b))

        if not gmpy2.is_finite(infVal):
            infVal = gmpy2.next_above(infVal)

        if not gmpy2.is_finite(supVal):
            supVal = gmpy2.next_below(supVal)

        sums = []
        # test if  the input is scalar or an array
        if np.isscalar(t):
            tt = []
            tt.append(t)
        else:
            tt = t
        # main loop through all floating point numbers in reduced precision
        countInitial = 0
        countFinal = 0
        for ti in tt:
            sum = 0.0
            err = float(ti) * eps

            x = mpfr(printMPFRExactly(infVal))
            y = gmpy2.next_above(x)
            z = gmpy2.next_above(y)

            xmin = (float(self.getInitialMinValue(infVal)) + float(printMPFRExactly(x))) / 2
            xmax = (float(printMPFRExactly(x)) + float(printMPFRExactly(y))) / 2.0
            xp = float(printMPFRExactly(x)) / (1.0 - err)
            if xmin < xp < xmax:
                sum += self.inputdistribution.get_piecewise_pdf()(xp) * abs(xp) * eps / (1.0 - err)
            # Deal with all standard intervals
            if y < supVal:
                while y < supVal:
                    xmin = xmax
                    xmax = (float(printMPFRExactly(y)) + float(printMPFRExactly(z))) / 2.0
                    xp = float(printMPFRExactly(y)) / (1.0 - err)

                    if xmin < xp < xmax:
                        sum += self.inputdistribution.get_piecewise_pdf()(xp) * abs(xp) * eps / (1.0 - err)

                    y = z
                    z = gmpy2.next_above(z)
                # Deal with the very last interval [x,(x+y)/2]
                # Z now should be equal to SUPVAL
                xmin = xmax
                xmax = (float(printMPFRExactly(y)) + float(self.getFinalMaxValue(supVal))) / 2.0
                xp = float(printMPFRExactly(y)) / (1.0 - err)

                # xp=mpfr(str(y))/(1.0-err)
                # xmax = mpfr(str(y))
                if xmin < xp < xmax:
                    sum += self.inputdistribution.get_piecewise_pdf()(xp) * abs(xp) * eps / (1.0 - err)

            sums.append(sum)

        reset_default_precision()

        if np.isscalar(t):
            return sum
        else:
            return sums


###
# Approximate Error Model given by the "Typical Distribution"
###

class TypicalError(ErrorModel):
    """
    An implementation of the typical error distribution with three segments
    """

    def __init__(self, input_distribution, precision=None, **kwargs):
        super(TypicalError, input_distribution, precision, None, self).__init__(**kwargs)
        self.name = "TypicalError(" + input_distribution.name + ")"
        self.p = precision

    def _left_segment(self, x):
        if self.p is None:
            y = 0.5 * ((1.0 / x) - 1.0) + 0.25 * (((1.0 / x) - 1.0) ** 2)
        else:
            u = 2 ** (-self.p - 1)
            alpha = np.floor(2 ** self.p * (-1 / x - 1) + 0.5)
            y = 1 / (2 ** self.p * (1 - u * x) ** 2) * (
                    2 / 3 + 0.5 * alpha + 2 ** (-self.p - 2) * alpha * (alpha - 1))
        return y

    def _middle_segment(self, t):
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

    def init_piecewise_pdf(self):
        self.piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        self.piecewise_pdf.addSegment(Segment(-1, -0.5, wrapped_pdf))
        self.piecewise_pdf.addSegment(Segment(-0.5, 0.5, wrapped_pdf))
        self.piecewise_pdf.addSegment(Segment(0.5, 1, wrapped_pdf))

    def rand_raw(self, n=None):  # None means return scalar
        inv_cdf = self.get_piecewise_invcdf()
        u = np.random.uniform(size=n)
        return inv_cdf(u)


###
# Wrapper class for all Error Models
###

class ErrorModelWrapper:
    """
    Wrapper class. ErrorModel is a PaCal object, ErrorModelWrapper is not.
    """
    def __init__(self, error_model):
        self.error_model = error_model
        self.sampleInit = True
        self.eps = 2 ** (-self.precision)
        self.distribution = error_model

    def createErrorDistr(self):
        self.distribution.get_piecewise_pdf()
        return self.distribution

    def execute(self):
        return self.distribution

    def getSampleSet(self, n=100000):
        # it remembers values for future operations
        if self.sampleInit:
            self.sampleSet = self.distribution.rand(n)
            self.sampleInit = False
        return self.sampleSet



class ErrorModelPointMass:
    def __init__(self, wrapperInputDistribution, precision, exp):
        self.wrapperInputDistribution = wrapperInputDistribution
        self.inputdistribution = self.wrapperInputDistribution.execute()
        self.inputdistribution.get_piecewise_pdf()
        self.precision = precision
        self.exp = exp
        self.eps = 2 ** -self.precision
        set_context_precision(self.precision, self.exp)
        qValue = printMPFRExactly(mpfr(str(self.inputdistribution.rand(1)[0])))
        reset_default_precision()
        error = float(str(self.inputdistribution.rand(1)[0])) - float(qValue)
        self.distribution = ConstDistr(float(error))
        self.distribution.get_piecewise_pdf()

    def execute(self):
        return self.distribution


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


def test_LP_error_model():
    exponent = 3
    mantissa = 4
    poly_precision = 200
    t = time()
    D = model.U("U", 2, 4)
    E = ErrorModel(D, mantissa, exponent, poly_precision)
    print(E.distribution.int_error())
    print(E.compare())
    print(time() - t)
    t = time()
    D = model.N("N", 0, 1)
    E = ErrorModel(D, mantissa, exponent, poly_precision)
    print(E.distribution.int_error())
    print(E.compare())
    print(time() - t)
    t = time()
    D = model.B("Beta", 2, 2)
    E = ErrorModel(D, mantissa, exponent, poly_precision)
    print(E.distribution.int_error())
    print(E.compare())
    print(time() - t)
    exponent = 5
    mantissa = 11
    poly_precision = 50
    t = time()
    D = model.U("U", 2, 4)
    E = ErrorModel(D, mantissa, exponent, poly_precision)
    print(E.distribution.int_error())
    print(E.compare())
    print(time() - t)
    t = time()
    D = model.N("N", 0, 1)
    E = ErrorModel(D, mantissa, exponent, poly_precision)
    print(E.distribution.int_error())
    print(E.compare())
    print(time() - t)
    t = time()
    D = model.B("Beta", 2, 2)
    E = ErrorModel(D, mantissa, exponent, poly_precision)
    print(E.distribution.int_error())
    print(E.compare())
    print(time() - t)



###
# OLD CODE
###

my_pdf = None


def genericPdf(x):
    if isinstance(x, float) or isinstance(x, int) or len(x) == 1:
        if x < -1 or x > 1:
            return 0
        else:
            return my_pdf(x)
    else:
        res = np.zeros(len(x))
        for index, ti in enumerate(x):
            if ti < -1 or ti > 1:
                res[index] = 0
            else:
                res[index] = my_pdf(ti)
        return res
    exit(-1)

def getTypical(x):
    if isinstance(x, float) or isinstance(x, int) or len(x) == 1:
        if abs(x) <= 0.5:
            return 0.75
        else:
            return 0.5 * ((1.0 / x) - 1.0) + 0.25 * (((1.0 / x) - 1.0) ** 2)
    else:
        res = np.zeros(len(x))
        for index, ti in enumerate(x):
            if abs(ti) <= 0.5:
                res[index] = 0.75
            else:
                res[index] = 0.5 * ((1.0 / ti) - 1.0) + 0.25 * (((1.0 / ti) - 1.0) ** 2)
        return res
    exit(-1)


typVariable = None


def createTypical(x):
    return typVariable(x)


class WrappedPiecewiseTypicalError():

    def __init__(self, p=None):
        self.sampleInit = True
        self.distribution = PiecewiseTypicalError(p)
        self.distribution.init_piecewise_pdf()

    def execute(self):
        return self.distribution

    def getSampleSet(self, n=100000):
        # it remembers values for future operations
        if self.sampleInit:
            self.sampleSet = self.distribution.rand(n)
            self.sampleInit = False
        return self.sampleSet


class TypicalErrorModel:
    def __init__(self, precision, exp, poly_precision):
        self.poly_precision = poly_precision
        self.name = "E"
        self.precision = precision
        self.exp = exp
        self.sampleInit = True
        self.eps = 2 ** (-self.precision)
        self.distribution = self.createTypicalErrorDistr()

    def createTypicalErrorDistr(self):
        global typVariable
        typVariable = chebfun(getTypical, domain=[-1.0, 1.0], N=self.poly_precision)
        self.distribution = FunDistr(createTypical, breakPoints=[-1.0, 1.0], interpolated=True)
        self.distribution.get_piecewise_pdf()
        return self.distribution

    def execute(self):
        return self.distribution

    def getSampleSet(self, n=100000):
        # it remembers values for future operations
        if self.sampleInit:
            self.sampleSet = self.distribution.rand(n)
            self.sampleInit = False
        return self.sampleSet


