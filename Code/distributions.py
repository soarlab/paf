from pacal.distr import *
from numpy import floor, ceil

# Classes with amend / customize PaCal classes

class SineDistr(FuncNoninjectiveDistr):
    """Sine of a random variable"""

    def __init__(self, d):
        """
        :param d: MUST be a PaCal distribution
        """
        self.base_distribution = d
        self. _get_intervals()
        self.fs = [numpy.sin] * len(self.intervals)
        self.f_invs = [numpy.arcsin, _pi_m_arcsin]
        self.f_inv_derivs = [_arcsin_der1, _arcsin_der1]
        self.pole_at_zero = False
        super(SineDistr, self).__init__(d, fname="sin")

    def _get_intervals(self):
        """
        :return: Generates a decomposition of the real line in intervals on which the sine function is monotone
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
        k = 0
        # first interval
        up = (2 * ceil(a/pi - 0.5) + 1) * pi / 2
        self.intervals.append([a, up])
        # regular intervals
        while up < b:
            down = up
            up += pi
            self.intervals.append([down, up])
        # last intervals
        down = (2 * floor(b/pi - 0.5) + 1) * pi / 2
        self.intervals.append([down, b])


def sin(d):
    """Overload the exp function."""
    if isinstance(d, Distr):
        return SineDistr(d)
    return numpy.sin(d)