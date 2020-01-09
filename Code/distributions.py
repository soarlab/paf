# Classes which re-implement or customize PaCal classes

import warnings
import numpy

from pacal.distr import FuncNoninjectiveDistr
from pacal.standard_distr import *
from pacal.vartransforms import VarTransform
from pacal.integration import _integrate_with_vartransform, integrate_fejer2
from numpy import ceil, log, arccos, arcsin, float_power
from numpy import isinf, isposinf, isfinite
from numpy import finfo, float32


def _shifted_arccos(x, shift):
    return arccos(x) + shift


def _shifted_arcsin(x, shift):
    return arcsin(x) + shift


def _arcsin_der(x):
    return (1 - x ** 2) ** (-0.5)


def _arccos_der(x):
    return -(1 - x ** 2) ** (-0.5)


def _strict_ceil(x):
    if x == ceil(x):
        return x + 1
    else:
        return ceil(x)


def integrate_fejer2_pinf_exp(f, log_a, log_b=None, *args, **kwargs):
    """Fejer2 integration from a to +oo."""
    if isposinf(log_a):
        return 0, 0
    vt = VarTransformExp_PInf(log_a, log_U=log_b)
    return _integrate_with_vartransform(f, vt, integrate_fejer2, *args, **kwargs)


class VarTransformExp_PInf(VarTransform):
    """Exponential variable transform.
    """

    def __init__(self, log_L=0, log_U=None):
        """
        :param L: MUST be the LOG of the lower bound of the integral to avoid instabilities
        :param U: MUST be the LOG of the lower bound of the integral to avoid instabilities
        """
        self.var_min = log_L
        if log_U is None:
            # We replace infinity by the log of the largest possible representable number
            self.var_max = log(finfo(float).max)
        else:
            self.var_max = log_U
        self.var_inf = [0]  # parameter values corresponding to infinity.  Do
        # not distinguish +oo and -oo

    def var_change(self, x):
        return log(x)

    def inv_var_change(self, y):
        return exp(y)

    def inv_var_change_deriv(self, y):
        return exp(y)


class PInfExpSegment(PInfSegment):
    """Segment = (a, inf]. Only the integrate method is overridden from PInfSegment
    """

    def __init__(self, a, f):
        super(PInfExpSegment, self).__init__(a, f)

    def integrate(self, log_a=None, log_b=None):
        if log_a is None or exp(log_a) < self.a:
            log_a = log(self.a)
        if log_b is None or isposinf(log_b):
            i, e = integrate_fejer2_pinf_exp(self.f, log_a)
        elif log_b > log_a:
            i, e = integrate_fejer2_pinf_exp(self.f, log_a, log_b)
        else:
            i, e = 0, 0
        return i


class ExpDistr(Distr):
    """Exponent of a random variable"""

    def __init__(self, d):
        """
        :param d: MUST be a PaCal distribution
        """
        self.base_distribution = d
        # Check whether the 1/t term causes a singularity at 0
        self.singularity_at_zero = self._detect_singularity()
        super(ExpDistr, self).__init__()

    def _exp_pdf(self, x):
        return self.base_distribution.get_piecewise_pdf()(log(x)) / x

    def _exp_out_of_range(self, x):
        if log(finfo(float).max) < x:
            return True
        return False

    def init_piecewise_pdf(self):
        self.piecewise_pdf = PiecewiseDistribution([])
        # Get the segments of base_distribution
        segs = self.base_distribution.get_piecewise_pdf().getSegments()
        if self._exp_out_of_range(segs[0].safe_a):
            raise ValueError('The smallest value in the support of the input distribution is too large. Exp not '
                             'supported')
        # Start with the possible problem at 0. The first segment will go from 0 to the min of
        # C and exp(segs[0].b)
        C = 0.1
        if log(C) < segs[0].safe_b:
            b = C
        else:
            b = exp(segs[0].safe_b)
        if self.singularity_at_zero:
            # test if the first segment of base_distribution has a right_pole
            if isinstance(segs[0], SegmentWithPole) and b < C:
                if ~segs[0].left_pole:
                    self.piecewise_pdf.addSegment(SegmentWithPole(1, exp(b / 2), self._exp_pdf, left_pole=True))
                    self.piecewise_pdf.addSegment(SegmentWithPole(exp(b / 2), exp(segs[0].safe_b), self._exp_pdf, left_pole=False))
            else:
                self.piecewise_pdf.addSegment(SegmentWithPole(1, exp(b), self._exp_pdf, left_pole=True))
                if b < segs[0].safe_b:
                    if self._exp_out_of_range(segs[0].safe_b):
                        warnings.warn(
                            "The support of exp(" + self.base_distribution.getName() + ") exceeds the range of "
                                                                                       "representable numbers. A "
                                                                                       "sub-distribution will be "
                                                                                       "constructed")
                        self.piecewise_pdf.addSegment(PInfExpSegment(exp(b), self._exp_pdf))
                        return
                    else:
                        self.piecewise_pdf.addSegment(Segment(exp(b), exp(segs[0].safe_b), self._exp_pdf))
        else:
            if self._exp_out_of_range(segs[0].safe_b):
                warnings.warn("The support of exp(" + self.base_distribution.getName() + ") exceeds the range of "
                                                                                         "representable numbers. A "
                                                                                         "sub-distribution will be "
                                                                                         "constructed")
                self.piecewise_pdf.addSegment(PInfExpSegment(exp(segs[0].safe_a), self._exp_pdf))
                return
            else:
                self.piecewise_pdf.addSegment(Segment(exp(segs[0].safe_a), exp(segs[0].safe_b), self._exp_pdf))
        if len(segs) > 1:
            for i in range(1, len(segs)):
                if self._exp_out_of_range(segs[i].safe_b):
                    self.piecewise_pdf.addSegment(PInfExpSegment(exp(segs[i].safe_a), self._exp_pdf))
                    return
                else:
                    self.piecewise_pdf.addSegment(Segment(exp(segs[i].safe_a), exp(segs[i].safe_b), self._exp_pdf))

    def _detect_singularity(self):
        """
        :return: A boolean value. True if pdf(ln(t))/t diverges at 0, False else. Divergence here is defined by:
        pdf(ln(t)) < t ** (1 + params.pole_detection.max_pole_exponent) for a sequence of small values starting at
        the smallest positive normal number in single precision (2**-126)
        """
        # Test if t can ever get close to 0. For this, the pdf must be defined at ln(small t), i.e. at sufficiently
        # negative numbers. We choose -40 as the cutoff.
        if isinf(self.base_distribution.range_()[0]):
            # Now test for divergence.
            for i in range(50):
                u = self.base_distribution.get_piecewise_pdf()(log(2 ** (-126 + i)))
                v = float_power(float_power(2, -126 + i), 1 + params.pole_detection.max_pole_exponent)
                if u > v:
                    return True
        elif self.base_distribution.range_()[0] < -40:
            # Now test for divergence. Here we start at min(2**-126, exp(self.base_distribution.range(0)))
            s = max(-126, self.base_distribution.range_()[0])
            for i in range(50):
                u = self.base_distribution.get_piecewise_pdf()(log(float(2 ** (s + i))))
                v = float_power(float_power(2, float(s + i)), 1 + params.pole_detection.max_pole_exponent)
                if u > v:
                    return True
        return False


class CosineDistr(FuncNoninjectiveDistr):
    """Cosine of a random variable"""

    def __init__(self, d):
        """
        :param d: MUST be a PaCal distribution
        """
        self.base_distribution = d
        self._get_intervals()
        self.pole_at_zero = False
        super(CosineDistr, self).__init__(d, fname="cos")

    def _get_intervals(self):
        """
        :return: Generates a decomposition of the real line in intervals on which the cosine function is monotone.
        On each interval the function (fs), its local inverse (f_invs), and the derivative of the local inverse
         (f_int_derivs) are recorded.
        """
        if isfinite(self.base_distribution.range_()[0]):
            a = self.base_distribution.range_()[0]
        # else we truncate the range of distribution so as to remove just a small amount of mass
        else:
            a = self.base_distribution.quantile(finfo(float32).eps)
        if isfinite(self.base_distribution.range_()[-1]):
            b = self.base_distribution.range_()[-1]
        else:
            b = self.base_distribution.quantile(1 - finfo(float32).eps)
        # Generate the intervals [k*pi, (k+1)*pi[ on which the cosine function is monotone
        self.intervals = []
        self.fs = []
        self.f_invs = []
        self.f_inv_derivs = []
        down = a
        k = _strict_ceil(a / pi - 1)
        up = a
        while up < b:
            if b < (k + 1) * pi:
                up = b
            else:
                up = (k + 1) * pi
            self.intervals.append([down, up])
            self.fs.append(cos)
            if k % 2 == 0:
                self.f_invs.append(partial(_shifted_arccos, shift=k * pi))
                self.f_inv_derivs.append(_arccos_der)
            else:
                self.f_invs.append(partial(_shifted_arcsin, shift=(2 * k + 1) * pi / 2))
                self.f_inv_derivs.append(_arcsin_der)
            k += 1
            down = up


class SineDistr(FuncNoninjectiveDistr):
    """Sine of a random variable"""

    def __init__(self, d):
        """
        :param d: MUST be a PaCal distribution
        """
        self.base_distribution = d
        self._get_intervals()
        self.pole_at_zero = False
        super(SineDistr, self).__init__(d, fname="sin")

    def _get_intervals(self):
        """
        :return: Generates a decomposition of the real line in intervals on which the sine function is monotone.
        On each interval the function (fs), its local inverse (f_invs), and the derivative of the local inverse
         (f_int_derivs) are recorded.
        """
        if isfinite(self.base_distribution.range_()[0]):
            a = self.base_distribution.range_()[0]
        # else we truncate the range of distribution so as to remove just a small amount of mass
        else:
            a = self.base_distribution.quantile(finfo(float32).eps)
        if isfinite(self.base_distribution.range_()[-1]):
            b = self.base_distribution.range_()[-1]
        else:
            b = self.base_distribution.quantile(1 - finfo(float32).eps)
        # Generate the intervals [(2k-1)*pi/2, (2k+1)*pi/2[ on which the sine function is monotone
        self.intervals = []
        self.fs = []
        self.f_invs = []
        self.f_inv_derivs = []
        down = a
        k = _strict_ceil(a / pi - 0.5)
        up = a
        while up < b:
            if b < (2 * k + 1) * pi / 2:
                up = b
            else:
                up = (2 * k + 1) * pi / 2
            self.intervals.append([down, up])
            self.fs.append(sin)
            if k % 2 == 1:
                self.f_invs.append(partial(_shifted_arccos, shift=(2 * k - 1) * pi / 2))
                self.f_inv_derivs.append(_arccos_der)
            else:
                self.f_invs.append(partial(_shifted_arcsin, shift=k * pi))
                self.f_inv_derivs.append(_arcsin_der)
            k += 1
            down = up


def sin(d):
    """Overload the sin function."""
    if isinstance(d, Distr):
        return SineDistr(d)
    return numpy.sin(d)


def cos(d):
    """Overload the sin function."""
    if isinstance(d, Distr):
        return CosineDistr(d)
    return numpy.cos(d)


def exp(d):
    """Overload the exp function."""
    if isinstance(d, Distr):
        return ExpDistr(d)
    return numpy.exp(d)


def testExp():
    X = UniformDistr(0, 700)
    expX = exp(X)
    print(expX.summary())
    Y = UniformDistr(0, 1000)
    expY = exp(Y)
    print(expY.summary())
    Z = NormalDistr()
    expZ = exp(Z)
    print(expZ.summary())
    W = BetaDistr(0.5,0.5)
    expW = exp(W)
    print(expW.summary())


def testCos():
    # this should agree with the PaCal implementation
    X = UniformDistr(0, pi)
    cosX = cos(X)
    print(cosX.summary())
    # these should show a huge improvement
    Y = UniformDistr(0, 20)
    cosY = cos(Y)
    print(cosY.summary())
    Z = NormalDistr()
    cosZ = cos(Z)
    print(cosZ.summary())


def testSin():
    # this should agree with the PaCal implementation
    X = UniformDistr(-pi / 2, pi / 2)
    sinX = sin(X)
    print(sinX.summary())
    # these should show a huge improvement
    Y = UniformDistr(0, 20)
    sinY = sin(Y)
    print(sinY.summary())
    Z = NormalDistr()
    sinZ = sin(Z)
    print(sinZ.summary())
