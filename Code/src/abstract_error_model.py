import matplotlib
import numpy as np
from gmpy2 import mpfr
from numpy import isscalar, zeros_like, asfarray
from pacal.distr import Distr
from pacal.segments import PiecewiseDistribution, Segment
import matplotlib.pyplot as plt
from pacal.utils import wrap_pdf
from scipy.stats import kstest
from project_utils import setCurrentContextPrecision, resetContextDefault

###
# Abstract ErrorModel class.
###
class AbstractErrorModel(Distr):
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
        super(AbstractErrorModel, self).__init__()
        self.input_distribution = input_distribution
        if input_distribution is None:
            self.name = "ErrorModel"
        else:
            self.name = "Error(" + input_distribution.getName() + ")"
            if self.input_distribution.execute() is None:
                self.input_distribution.execute().get_piecewise_pdf()
        # gmpy2 precision (or None):
        self.precision = precision
        self.polynomial_precision = polynomial_precision
        self.exponent = exponent
        # In gmpy2 precision includes a sign bit, so 2 ** precision = unit roundoff
        if self.precision is None:
            self.unit_roundoff = 2 ** (-24)
        else:
            self.unit_roundoff = 2 ** (-self.precision)

    #def init_piecewise_pdf(self):
    #    """Initialize the pdf represented as a piecewise function.
    #    This method should be overridden by subclasses."""
    #    raise NotImplementedError()

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

    def execute(self):
        self.init_piecewise_pdf()
        return self

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
        setCurrentContextPrecision(self.precision, self.exponent)
        for index, ti in enumerate(empirical):
            rounded[index] = mpfr(str(empirical[index]))
        resetContextDefault()
        empirical = (empirical - rounded) / (empirical * self.unit_roundoff)
        ks_test = kstest(empirical, cdf)
        x = np.linspace(-1, 1, 201)
        matplotlib.pyplot.close("all")
        matplotlib.rcParams.update({'font.size': 12})
        plt.hist(empirical, bins=2 * np.floor(n ** (1 / 3)), range=[-1, 1], density=True)
        y = pdf(x)
        h = plt.plot(x, y)
        plt.title(
            self.input_distribution.getName() + ", KS-test=" + str(round(ks_test[0], 4)) + ", p-val=" + str(round(ks_test[1], 4)))
        if file_name is None:
            plt.show()
        else:
            plt.savefig("file_name" + self.getName() + ".png")
        return ks_test
