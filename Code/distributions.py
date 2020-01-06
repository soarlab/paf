from pacal.distr import *
from numpy import floor, ceil

# Classes with amend / customize PaCal classes


def _shifted_arccos(x, shift):
    return arccos(x) + shift


def _shifted_arcsin(x, shift):
    return arcsin(x) + shift


def _arcsin_der(x):
    return (1-x**2)**(-0.5)


def _arccos_der(x):
    return -(1-x**2)**(-0.5)


class SineDistr(FuncNoninjectiveDistr):
    """Sine of a random variable"""

    def __init__(self, d):
        """
        :param d: MUST be a PaCal distribution
        """
        self.base_distribution = d
        self. _get_intervals()
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
        if isfinite(self.distribution.range_()[-1]):
            b = self.distribution.range_()[-1]
        else:
            b = self.base_distribution.quantile(1-numpy.finfo(numpy.float32).eps)
        # Generate the monotone intervals [(2k-1)*pi/2, (2k+1)*pi/2]
        self.intervals = []
        self.fs = []
        self.f_invs = []
        self.f_inv_derivs = []
        # first interval
        k = ceil(a/pi - 0.5)
        up = (2*k + 1) * pi / 2
        self.intervals.append([a, up])
        self.fs.append(numpy.sin)
        if k % 2 == 0:
            self.f_invs.append(partial(_shifted_arccos, shift=(2*k - 1) * pi / 2))
            self.f_inv_derivs.append(_arccos_der)
        else:
            self.f_invs.append(partial(_shifted_arcsin, shift=k*pi))
            self.f_inv_derivs.append(_arcsin_der)
        # regular intervals
        while up < b:
            down = up
            up += pi
            k += 1
            self.intervals.append([down, up])
            self.fs.append(numpy.sin)
            if k % 2 == 0:
                self.f_invs.append(partial(_shifted_arccos, shift=(2 * k - 1) * pi / 2))
                self.f_inv_derivs.append(_arccos_der)
            else:
                self.f_invs.append(partial(_shifted_arcsin, shift=k * pi))
                self.f_inv_derivs.append(_arcsin_der)
        # last interval
        self.intervals.append([down, b])
        self.fs.append(numpy.sin)
        if k % 2 == 0:
            self.f_invs.append(partial(_shifted_arccos, shift=(2*k - 1) * pi / 2))
            self.f_inv_derivs.append(_arccos_der)
        else:
            self.f_invs.append(partial(_shifted_arcsin, shift=k*pi))
            self.f_inv_derivs.append(_arcsin_der)


def sin(d):
    """Overload the exp function."""
    if isinstance(d, Distr):
        return SineDistr(d)
    return numpy.sin(d)