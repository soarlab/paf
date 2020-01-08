from pacal.distr import *
from pacal.standard_distr import *
from numpy import ceil, isinf, log, inf
import scipy.integrate as integrate

# Classes which re-implement or customize PaCal classes


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


class ExpDistr(Distr):
    """Exponential of a distribution"""

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

    def _interpolate(self, a, b):
        """ Segment gets extremely large extremely fast because of the presence of the exp function,
        and the accuracy of interpolations suffers accordingly. This method tries to solve this problem
        by recursively partitioning segments into smaller segments logarithmically
         :param a,b: MUST be in log form
         """
        error = 1
        N = 1
        p_func = PiecewiseFunction([])
        while error > 0.01:
            # Generate logarithmically spaced breakpoints
            breakpoints = []
            for i in range(0, N+1):
                breakpoints.append(a + i*(b-a)/N)
            for i in range(0, N):
                if i == N-1:
                    p_func.addSegment(PInfSegment(exp(breakpoints[i]), self._exp_pdf))
                else:
                    p_func.addSegment(Segment(exp(breakpoints[i]), exp(breakpoints[i+1]), self._exp_pdf))
            i_func = p_func.toInterpolated()
            N += 1
        return p_func

    def _interpolate_to_infinity(self, a):
        """ Same as above but to infinity"""
        error = 1
        N = 1
        p_func = PiecewiseFunction([])
        while error < 0.01:
            # Generate logarithmically spaced breakpoints
            breakpoints = []
            for i in range(0, N):
                breakpoints[i] = (10**i)*a
            for i in range(0, N - 1):
                if i == N - 1:
                    p_func.addSegment(PInfSegment(breakpoints[i], self._exp_pdf))
                else:
                    p_func.addSegment(Segment(breakpoints[i], breakpoints[i + 1], self._exp_pdf))
            error = integrate.quad(lambda x: abs(p_func(x) - self._exp_pdf(x)), a, inf)
            N += 1
        return p_func

    def init_piecewise_pdf(self):
        if self._exp_out_of_range(self.base_distribution.range_()[0]):
            raise ValueError('The smallest value in the support of the input distribution is too large. Exp not '
                             'supported')
        if isinf(self.base_distribution.range_()[0]):
            self.a = 0
        else:
            self.a = exp(self.base_distribution.range_()[0])
        self.piecewise_pdf = PiecewiseDistribution([])
        # Get the segments of base_distribution
        segs = self.base_distribution.get_piecewise_pdf().getSegments()
        # Start with the possible problem at 0. The first segment will go from 0 to the min of
        # 0.5 and segs[0].b
        if log(0.5) < segs[0].safe_b:
            b = 0.5
        else:
            b = exp(segs[0].safe_b)
        if self.singularity_at_zero:
            # test if the first segment of base_distribution has a right_pole
            if isinstance(segs[0].SegmentWithPole) and b < 0.5:
                if ~segs[0].left_pole:
                    self.piecewise_pdf.addSegment(SegmentWithPole(0, b/2, self._exp_pdf, left_pole=True))
                    self.piecewise_pdf.addSegment(SegmentWithPole(b/2, b, self._exp_pdf, left_pole=False))
            else:
                self.piecewise_pdf.addSegment(SegmentWithPole(0, b, self._exp_pdf, left_pole=True))
        else:
            if self._exp_out_of_range(segs[0].safe_b):
                if isinf(segs[0].safe_b):
                    self._interpolate_to_infinity(segs[0].safe_a)
                else:
                    self._interpolate(segs[0].safe_a, segs[0].safe_b)
                return
            else:
                self.piecewise_pdf.addSegment(Segment(self.a, exp(segs[0].safe_b), self._exp_pdf))
        if len(segs) > 1:
            for i in range(1, len(segs)):
                if self._exp_out_of_range(segs[i].safe_b):
                    self._fill_to_infty(self, i)
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
                if self.base_distribution.get_piecewise_pdf()(log(2 ** (-126 + i))) > 2 ** (-126 + i) ** (
                        1 + params.pole_detection.max_pole_exponent):
                    return True
        elif self.base_distribution.range_()[0] < -40:
            # Now test for divergence. Here we start at min(2**-126, exp(self.base_distribution.range(0)))
            s = max(-126, self.base_distribution.range_()[0])
            for i in range(50):
                if self.base_distribution.get_piecewise_pdf()(log(2 ** (-s + i))) > 2 ** (-s + i) ** (
                        1 + params.pole_detection.max_pole_exponent):
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
            a = self.base_distribution.quantile(numpy.finfo(numpy.float32).eps)
        if isfinite(self.base_distribution.range_()[-1]):
            b = self.base_distribution.range_()[-1]
        else:
            b = self.base_distribution.quantile(1 - numpy.finfo(numpy.float32).eps)
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
            self.fs.append(numpy.cos)
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
            a = self.base_distribution.quantile(numpy.finfo(numpy.float32).eps)
        if isfinite(self.base_distribution.range_()[-1]):
            b = self.base_distribution.range_()[-1]
        else:
            b = self.base_distribution.quantile(1 - numpy.finfo(numpy.float32).eps)
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
            self.fs.append(numpy.sin)
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
    X = UniformDistr(0, 1000)
    expX = exp(X)
    print(expX.summary())


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
