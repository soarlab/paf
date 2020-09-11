import matplotlib
import numpy as np
from gmpy2 import mpfr
from numpy import isscalar, zeros_like, asfarray
from pacal.distr import Distr
from pacal.segments import PiecewiseDistribution, Segment
import matplotlib.pyplot as plt
from math import floor
from pacal.utils import wrap_pdf
from scipy.stats import kstest
from pychebfun import Chebfun
from project_utils import set_context_precision, reset_default_precision


###
# Abstract ErrorModel class.
###
class AbstractErrorModel(Distr):
    def __init__(self, input_distribution, precision, exponent, polynomial_precision=[0, 0]):
        """
        Error distribution class.
        Inputs:
            :param input_distribution: a PaCal object representing the distribution for which we want to compute
                                the rounding error distribution
            :param precision, exponent: gmpy2 precision environment
            :param polynomial_precision: a pair of integers controlling the precision of the polynomial interpolation of
                                         the middle and wing segments respectively. Warning: this is not the number of
                                         interpolation points.
                                         Default is (0,0) which means that it is determined dynamically by pychebfun.
                                         0 would be a nonsensical value anyway, so it's safe to use it as a flag.
        """
        super(AbstractErrorModel, self).__init__()
        self.input_distribution = input_distribution
        # gmpy2 precision and precision:
        self.precision = precision
        self.exponent = exponent
        # In gmpy2 precision includes a sign bit, so 2 ** precision = unit roundoff
        self.unit_roundoff = 2 ** (-self.precision)
        self.eps = 2 ** (-self.precision)
        # numbers of interpolation points - if input doesn't make sense, set to default
        if not isinstance(polynomial_precision, list):
            self.polynomial_precision = [0, 0]
        elif len(polynomial_precision) != 2:
            self.polynomial_precision = [0, 0]
        else:
            self.polynomial_precision = polynomial_precision
        self.a=-self.eps
        self.b=self.eps
        self.name="Abstract_Error_Model"

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
        if self.polynomial_precision[0] == 0:
            seg1 = Chebfun.from_function(wrapped_pdf, [-1.0, -0.5])
        else:
            seg1 = Chebfun.from_function(wrapped_pdf, [-1.0, -0.5], self.polynomial_precision[0])
        if self.polynomial_precision[1] == 0:
            seg2 = Chebfun.from_function(wrapped_pdf, [-0.5, 0.5])
        else:
            seg2 = Chebfun.from_function(wrapped_pdf, [-0.5, 0.5], self.polynomial_precision[1])
        if self.polynomial_precision[0] == 0:
            seg3 = Chebfun.from_function(wrapped_pdf, [0.5, 1.0])
        else:
            seg3 = Chebfun.from_function(wrapped_pdf, [0.5, 1.0], self.polynomial_precision[0])
        piecewise_pdf.addSegment(Segment(-1.0, -0.5, seg1))
        piecewise_pdf.addSegment(Segment(-0.5, 0.5, seg2))
        piecewise_pdf.addSegment(Segment(0.5, 1.0, seg3))
        self.piecewise_pdf = piecewise_pdf

    def pdf(self, x):
        if isscalar(x):
            if -1.0 <= x < -0.5:
                return self._left_segment(x)
            elif -0.5 <= x <= 0.5:
                return self._middle_segment(x)
            elif 0.5 < x <= 1.0:
                return self._right_segment(x)
            else:
                return 0.0
        else:
            y = zeros_like(asfarray(x))
            for index, ti in enumerate(x):
                if -1.0 <= ti < -0.5:
                    y[index] = self._left_segment(ti)
                elif -0.5 <= ti <= 0.5:
                    y[index] = self._middle_segment(ti)
                elif 0.5 < ti <= 1.0:
                    y[index] = self._right_segment(ti)
            return y

    def execute(self):
        self.get_piecewise_pdf()
        return self

    def rand_raw(self, n=None):  # None means return scalar
        inv_cdf = self.get_piecewise_invcdf(use_interpolated=False)
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

    def compare(self, n=20000, file_name=None):
        """
        A function to compare the ErrorModel density function with an empirical distribution of relative errors
        and return a K-S test
        :param n: number of samples
        :param file_name: optional, if not None the graph will be saved using the file_name + name of the distribution.
        :return: the Kolmogorov-Smirnov (K-S) statistic and p-value
        """
        if self.input_distribution is None:
            return "Nothing to compare against!"
        # Typical distribution does not require a precision or exponent parameter, in that case choose single precision
        if self.precision is None:
            precision = 24
        else:
            precision = self.precision
        if self.exponent is None:
            exponent = 8
        else:
            exponent = self. exponent
        empirical = self.input_distribution.rand(n)
        pdf = self.get_piecewise_pdf()
        cdf = self.get_piecewise_cdf()
        rounded = np.zeros_like(empirical)
        set_context_precision(precision, exponent)
        for index, ti in enumerate(empirical):
            rounded[index] = mpfr(str(empirical[index]))
        reset_default_precision()
        empirical = (empirical - rounded) / (empirical * (2 ** -precision))
        ks_test = kstest(empirical, cdf)
        x = np.linspace(-1, 1, 201)
        matplotlib.pyplot.close("all")
        matplotlib.rcParams.update({'font.size': 12})
        plt.hist(empirical, bins=2 * floor(n ** (1 / 3)), range=[-1, 1], density=True)
        y = pdf(x)
        h = plt.plot(x, y)
        plt.title(
            self.input_distribution.getName() + ", KS-test=" + str(round(ks_test[0], 4)) + ", p-val=" + str(
                round(ks_test[1], 4)))
        if file_name is None:
            plt.show()
        else:
            plt.savefig("file_name" + self.getName() + ".png")
        return ks_test
