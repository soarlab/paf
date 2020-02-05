from pacal import *
import matplotlib.pyplot as plt
import numpy as np
import gmpy2
import numpy as np;
from time import time
from pychebfun import *
from utils import *
from gmpy2 import mpfr
from pacal.distr import Distr
from pacal.segments import PiecewiseDistribution, Segment
from pacal.utils import wrap_pdf
from numpy import isscalar, zeros_like, asfarray
from scipy import integrate
from scipy.stats import kstest
import dill
import pickle

import scipy
import random

##################################
#### How to use FunDistr #########
##################################
# Those need to be defined before PaCAL creates a process pool for
# picklability:
# def example_f1(x):
#    return sqrt(2)/numpy.pi / (1+x**4)
# def example_f2(x):
#    return 1.5*x*x
# f = FunDistr(example_f1 , [-Inf, -1, 0, 1, Inf])

import gmpy2
import math
import pacal
import matplotlib.pyplot as plt
import numpy as np


def setCurrentContextPrecision(mantissa, exponent):
    ctx = gmpy2.get_context()
    ctx.precision = mantissa
    ctx.emax = 2 ** (exponent - 1)
    ctx.emin = 1 - ctx.emax


def resetContextDefault():
    gmpy2.set_context(gmpy2.context())


class ErrorModelNaive:
    def __init__(self, distribution, precision, exp, samplesize):
        self.inputdistribution = distribution
        self.precision = precision
        self.exp = exp
        self.samplesize = samplesize
        # self.distribution=self.compute_naive_error()

    def compute_naive_error(self):
        x = self.inputdistribution.rand(self.samplesize)
        # x=list( dict.fromkeys(x))
        self.samplesize = len(x)
        errors = []
        eps = 2 ** -self.precision

        setCurrentContextPrecision(self.precision, self.exp)

        for r in x:
            # In this way 'e' is always scaled between 0 and 1.
            e = (r - float(gmpy2.round2(r, self.precision))) / (r * eps)
            errors.append(e)

        resetContextDefault()

        return x, errors

    def plot_error(self, errors, figureName):
        plt.figure(figureName)
        bin_nb = int(math.ceil(math.sqrt(self.samplesize)))
        n, bins, patches = np.histogram(errors, bins=bin_nb, range=(-1, +1), density=1, label="Naive")
        plt.legend()
        axes = plt.gca()
        axes.set_xlim([-1, 1])
        return
        # plt.savefig('pics/unifsmall_'+repr(precision))
        # plt.savefig('pics/'+repr(distribution.getName()).replace("'",'')+'_'+repr(precision))
        # plt.clf()


class ErrorModelPointMass:
    def __init__(self, wrapperInputDistribution, precision, exp):
        self.wrapperInputDistribution = wrapperInputDistribution
        self.inputdistribution = self.wrapperInputDistribution.execute()
        self.inputdistribution.get_piecewise_pdf()
        self.precision = precision
        self.exp = exp
        self.eps = 2 ** -self.precision
        setCurrentContextPrecision(self.precision, self.exp)
        qValue = printMPFRExactly(mpfr(str(self.inputdistribution.rand(1)[0])))
        resetContextDefault()
        error = float(str(self.inputdistribution.rand(1)[0])) - float(qValue)
        self.distribution = ConstDistr(float(error))
        self.distribution.get_piecewise_pdf()

    def execute(self):
        self.distribution.init_piecewise_pdf()
        return self.distribution


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


class WrappedHighPrecisionError():

    def __init__(self, input_distribution, precision, exponent):
        self.name = "Error(" + input_distribution.name + ")"
        self.precision = precision
        self.exp = exponent
        self.sampleInit = True
        self.eps = 2 ** (-self.precision)
        self.distribution = HighPrecisionErrorModel(input_distribution, precision, exponent)

    def createErrorDistr(self):
        self.distribution.init_piecewise_pdf()
        return self.distribution

    def execute(self):
        return self.distribution

    def getSampleSet(self, n=100000):
        # it remembers values for future operations
        if self.sampleInit:
            self.sampleSet = self.distribution.rand(n)
            self.sampleInit = False
        return self.sampleSet


class HighPrecisionErrorModel(Distr):

    def __init__(self, input_distribution, precision, exp):
        '''
    The class implements the high-precision error distribution function.
    Inputs:
        input_distribution: a PaCal object representing the distribution for which we want to compute
                            the rounding error distribution
        precision, minexp, maxexp: specify the low precision environment suing gmpy2
        '''
        super(HighPrecisionErrorModel, self).__init__()
        self.input_distribution = input_distribution.execute()
        self.input_distribution.get_piecewise_pdf()
        self.name = "Error(" + input_distribution.name + ")"
        self.precision = precision
        self.exp = exp
        self.sampleInit = True
        self.central_constant = None
        self.eps = 2 ** (-self.precision)
        self._get_min_exponent()
        self._get_max_exponent()

    def getName(self):
        return self.name

    def init_piecewise_pdf(self):
        if self.central_constant is None:
            self._compute_central_constant()
        piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        piecewise_pdf.addSegment(Segment(-1, -0.5, wrapped_pdf))
        piecewise_pdf.addSegment(Segment(-0.5, 0.5, wrapped_pdf))
        piecewise_pdf.addSegment(Segment(0.5, 1, wrapped_pdf))
        self.piecewise_pdf = piecewise_pdf.toInterpolated()

    def range(self):
        return -1.0, 1.0

    def rand_raw(self, n=None):  # None means return scalar
        inv_cdf = self.get_piecewise_invcdf()
        u = np.random.uniform(size=n)
        return inv_cdf(u)

    def pdf(self, x):
        if isscalar(x):
            if abs(x) <= 0.5:
                return self._pdf_middle(x)
            elif abs(x) <= 1:
                return self._pdf_wing(x)
            else:
                return 0.0
        else:
            y = zeros_like(asfarray(x))
            for index, ti in enumerate(x):
                if abs(ti) <= 0.5:
                    y[index] = self._pdf_middle(ti)
                elif abs(ti) <= 1:
                    y[index] = self._pdf_wing(ti)
            return y

    def compare(self, n=100000):
        """A function to compare the density function with a Monte-Carlo simulation and return a K-S test"""
        empirical = self.input_distribution.rand(n)
        f = self.get_piecewise_pdf()
        rounded = np.zeros_like(empirical)
        setCurrentContextPrecision(self.precision, self.exp)
        for index, ti in enumerate(empirical):
            rounded[index] = mpfr(str(empirical[index]))
        resetContextDefault()
        for index, ti in enumerate(empirical):
            x = (ti - rounded[index]) / (ti * self.eps)
            if is_finite(x) and ~is_nan(x):
                empirical[index] = x
            else:
                empirical[index] = 0
        KS = kstest(empirical, f)
        x = np.linspace(-1, 1, 201)
        plt.close()
        plt.hist(empirical, bins=math.floor(n ** (1 / 3)), range=[-1, 1], density=True)
        y = f(x)
        h = plt.plot(x, y)
        plt.show()
        return KS

    def _pdf_wing(self, x):
        """
        :param x: SCALAR real such that 0.5 < abs(x) <= 1. Arrays are dealt with in self.pdf.
        :return: pdf evaluated at (x)
        """
        return self._compute_integral(x) / ((1 - self.eps * x) ** 2)

    def _pdf_middle(self, x):
        """
        :param x: SCALAR real such that abs(x) <= 0.5. Arrays are dealt with in self.pdf.
        :return: pdf evaluated at (x)
        """
        return self.central_constant / ((1 - self.eps * x) ** 2)

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
        setCurrentContextPrecision(self.precision, self.exp)
        inf_val = mpfr(str(self.input_distribution.range_()[0]))
        self.min_sign = gmpy2.sign(inf_val)
        # For some reason the exponent returned by get_exp() is 1 too high and 0 for infinities
        if is_finite(inf_val):
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
        resetContextDefault()

    def _get_max_exponent(self):
        setCurrentContextPrecision(self.precision, self.exp)
        sup_val = mpfr(str(self.input_distribution.range_()[1]))
        self.max_sign = gmpy2.sign(sup_val)
        # For some reason the exponent returned by get_exp() is 1 too high and 0 if sup_val is infinite
        if is_finite(sup_val):
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
        resetContextDefault()


def test_error_model():
    t = time()
    U = BetaDistr(3, 2)
    E = HighPrecisionErrorModel(U, 23, 8)
    E.init_piecewise_pdf()
    print(E.getName())
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    t = time()
    U = UniformDistr(64, 1024)
    E = HighPrecisionErrorModel(U, 23, 8)
    print(E.getName())
    E.init_piecewise_pdf()
    print(E.int_error())
    print(time() - t)
    t = time()
    U = UniformDistr(-1024, -64)
    E = HighPrecisionErrorModel(U, 23, 8)
    print(E.getName())
    E.init_piecewise_pdf()
    print(E.int_error())
    print(time() - t)
    t = time()
    U = BetaDistr()
    E = HighPrecisionErrorModel(U, 23, 8)
    print(E.getName())
    E.init_piecewise_pdf()
    print(E.int_error())
    print(time() - t)
    U = NormalDistr()
    E = HighPrecisionErrorModel(U, 23, 8)
    print(E.getName())
    E.init_piecewise_pdf()
    print(E.int_error())
    print(time() - t)


class ErrorModel:

    def __init__(self, wrapperInputDistribution, precision, exp, poly_precision):
        '''
    Constructor interpolates the density function using Chebyshev interpolation
    then uses this interpolation to build a PaCal object:
    the self.distribution attribute which contains all the methods we could possibly want
    Inputs:
        inputdistribution: a PaCal object representing the distribution for which we want to compute
                            the rounding error distribution
        precision, minexp, maxexp: specify the low precision environment suing gmpy2
        poly_precision: the number of exact evaluations of the density function used to
                        build the interpolating polynomial representing it
        '''
        self.wrapperInputDistribution = wrapperInputDistribution
        self.inputdistribution = self.wrapperInputDistribution.execute()
        self.inputdistribution.get_piecewise_pdf()
        self.name = "E"
        self.precision = precision
        self.exp = exp
        self.sampleInit = True
        self.eps = 2 ** (-self.precision)

        self.poly_precision = poly_precision
        # Test if the range of floating point number covers enough of the inputdistribution
        x = gmpy2.next_above(gmpy2.inf(-1))
        y = gmpy2.next_below(gmpy2.inf(1))
        # check exponenent out of range (overflow)
        # instead in case of accuracy problem
        # (normalize: divide by the current coverage ex. 0.995/0.995)

        # coverage=self.inputdistribution.get_piecewise_pdf().integrate(float("-inf"),float("+inf"))
        # if coverage<0.99:
        #    raise Exception('The range of floating points is too narrow, increase maxexp and increase minexp')
        # Builds the Chebyshev polynomial representation of the density function

        global my_pdf
        my_pdf = chebfun(self.getpdf, domain=[-1.0, 1.0], N=self.poly_precision)
        self.distribution = FunDistr(genericPdf, breakPoints=[-1.0, 1.0], interpolated=True)
        # self.distribution = FunDistr(self.pdf.p, [-1.0, 1.0])
        self.distribution.init_piecewise_pdf()
        self.distribution.get_piecewise_pdf()

    def __call__(self, t):
        return ErrorModel.getpdf(self, t)

    # Quick and dirty plotting function
    def plot(self, strFile):
        x = np.linspace(-1, 1, 201)
        y = self.pdf(x)
        plt.plot(x, y)
        plt.savefig(strFile)
        plt.clf()

    def execute(self):
        return self.distribution

    def getSampleSet(self, n=100000):
        # it remembers values for future operations
        if self.sampleInit:
            self.sampleSet = self.distribution.rand(n)
            # self.sampleSet = self.distribution.rand(n - 2)
            # self.sampleSet = np.append(self.sampleSet, [-1.0, 1.0])
            self.sampleInit = False
        return self.sampleSet

    # infVal is finite value
    def getInitialMinValue(self, infVal):
        if not gmpy2.is_finite(infVal):
            print("Error cannot compute intervals with infinity")
            exit(-1)
        bkpCtx = gmpy2.get_context().copy()

        i = 0
        while not gmpy2.is_finite(gmpy2.next_below(infVal)):
            setCurrentContextPrecision(self.precision + i, self.exp + i)
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
            setCurrentContextPrecision(self.precision + i, self.exp + i)
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

        setCurrentContextPrecision(self.precision, self.exp)
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

        resetContextDefault()

        if np.isscalar(t):
            return sum
        else:
            return sums


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


class PiecewiseTypicalError(Distr):
    """
    An implementation of the typical error distribution with three segments
    """

    def __init__(self, p=None, **kwargs):
        super(PiecewiseTypicalError, self).__init__(**kwargs)
        self.p = p

    def pdf(self, x):
        if isscalar(x):
            if abs(x) > 1:
                y = 0
            else:
                if abs(x) <= 0.5:
                    if self.p is None:
                        y = 0.75
                    else:
                        u = 2 ** (-self.p - 1)
                        y = 1 / (2 ** self.p * (1 - u * x) ** 2) * (2 / 3 + 3 * (2 ** self.p - 1) / 4)
                else:
                    if self.p is None:
                        y = 0.5 * ((1.0 / x) - 1.0) + 0.25 * (((1.0 / x) - 1.0) ** 2)
                    else:
                        u = 2 ** (-self.p - 1)
                        if x > 0:
                            alpha = np.floor(2 ** self.p * (1 / x - 1) - 0.5)
                        else:
                            alpha = np.floor(2 ** self.p * (-1 / x - 1) + 0.5)
                        y = 1 / (2 ** self.p * (1 - u * x) ** 2) * (
                                2 / 3 + 0.5 * alpha + 2 ** (-self.p - 2) * alpha * (alpha - 1))
        else:
            y = zeros_like(asfarray(x))
            for index, ti in enumerate(x):
                if abs(ti) <= 0.5:
                    if self.p is None:
                        y[index] = 0.75
                    else:
                        u = 2 ** (-self.p - 1)
                        y[index] = 1 / (2 ** self.p * (1 - u * ti) ** 2) * (2 / 3 + 3 * (2 ** self.p - 1) / 4)
                elif abs(ti) <= 1.0:
                    if self.p is None:
                        y[index] = 0.5 * ((1.0 / ti) - 1.0) + 0.25 * (((1.0 / ti) - 1.0) ** 2)
                    else:
                        u = 2 ** (-self.p - 1)
                        if ti > 0:
                            alpha = np.floor(2 ** self.p * (1 / ti - 1) - 0.5)
                        else:
                            alpha = np.floor(2 ** self.p * (-1 / ti - 1) + 0.5)
                        y[index] = 1 / (2 ** self.p * (1 - u * ti) ** 2) * (
                                2 / 3 + 0.5 * alpha + 2 ** (-self.p - 2) * alpha * (alpha - 1))
        return y

    def _alpha(self, x):
        return floor(2 ** self.p * (1 / x - 1) - 0.5)

    def init_piecewise_pdf(self):
        self.piecewise_pdf = PiecewiseDistribution([])
        wrapped_pdf = wrap_pdf(self.pdf)
        self.piecewise_pdf.addSegment(Segment(-1, -0.5, wrapped_pdf))
        self.piecewise_pdf.addSegment(Segment(-0.5, 0.5, wrapped_pdf))
        self.piecewise_pdf.addSegment(Segment(0.5, 1, wrapped_pdf))

    def __str__(self):
        if self.p is None:
            return "Typical#{0}".format(self.id())
        else:
            return "Typical(p={0})#{1}".format(self.p, self.id())

    def getName(self):
        if self.p is None:
            return "Typical"
        else:
            return "Typical({0})".format(self.p)

    def range(self):
        return -1.0, 1.0

    def rand_raw(self, n=None):  # None means return scalar
        inv_cdf = self.get_piecewise_invcdf()
        u = np.random.uniform(size=n)
        return inv_cdf(u)


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
            #self.sampleSet  = self.distribution.rand(n-2)
            #self.sampleSet  = np.append(self.sampleSet, [-1.0, 1.0])
            # self.sampleSet = sorted(self.sampleSet)
            self.sampleInit = False
        return self.sampleSet
