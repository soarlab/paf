from pacal.distr import *
from pacal.standard_distr import  *
from numpy import ceil

# Classes which amend / customize PaCal classes


def _shifted_arccos(x, shift):
    return arccos(x) + shift


def _shifted_arcsin(x, shift):
    return arcsin(x) + shift


def _arcsin_der(x):
    return (1-x**2)**(-0.5)


def _arccos_der(x):
    return -(1-x**2)**(-0.5)


def _strict_ceil(x):
    if x == ceil(x):
        return x+1
    else:
        return ceil(x)


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
            b = self.base_distribution.quantile(1-numpy.finfo(numpy.float32).eps)
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
            b = self.base_distribution.quantile(1-numpy.finfo(numpy.float32).eps)
        # Generate the intervals [(2k-1)*pi/2, (2k+1)*pi/2[ on which the sine function is monotone
        self.intervals = []
        self.fs = []
        self.f_invs = []
        self.f_inv_derivs = []
        down = a
        k = _strict_ceil(a/pi-0.5)
        up = a
        while up < b:
            if b < (2*k+1)*pi/2:
                up = b
            else:
                up = (2*k+1)*pi/2
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
    X = UniformDistr(-pi/2, pi/2)
    sinX = sin(X)
    print(sinX.summary())
    # these should show a huge improvement
    Y = UniformDistr(0, 20)
    sinY = sin(Y)
    print(sinY.summary())
    Z = NormalDistr()
    sinZ = sin(Z)
    print(sinZ.summary())