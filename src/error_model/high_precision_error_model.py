
import gmpy2
from scipy import integrate

from .abstract_error_model import AbstractErrorModel

from project_utils import set_context_precision, reset_default_precision

###
# Exact Error Distribution for High-Precisions (above half-precision).
###
class HighPrecisionErrorModel(AbstractErrorModel):

    def __init__(self, input_distribution, input_name, precision, exponent, polynomial_precision):
        """
        The class implements the high-precision error distribution function.
        Inputs:
            input_distribution: a PaCal object representing the distribution for which we want to compute
                            the rounding error distribution
            precision, exponent: gmpy2 precision environment
        """
        super(HighPrecisionErrorModel, self).__init__(input_distribution, precision, exponent, polynomial_precision)
        #self.name = "HPError(" + input_distribution.getName() + ")"
        self.name = "HPE_" + input_name
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
        inf_val = gmpy2.mpfr(str(self.input_distribution.range_()[0]))
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
        sup_val = gmpy2.mpfr(str(self.input_distribution.range_()[1]))
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