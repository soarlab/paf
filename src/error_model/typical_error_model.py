from .abstract_error_model import AbstractErrorModel
import numpy as np
from math import floor

###
# Approximate Error Model given by the "Typical Distribution"
###

class TypicalErrorModel(AbstractErrorModel):
    """
    An implementation of the typical error distribution with three segments
    All inputs can be left unspecified.
    Inputs:
    polynomial_precision: should be left unspecified when the precision is unspecified, the typical distribution is so
                          regular that pychebfun always converges. This is not the case when the precision correction
                          is applied, so default is to then assign values dynamically
    precision_correction: if True specified a correction will be applied. Only makes a notable difference at very
                          low precision (< 7 bits)
    """
    def __init__(self, input_distribution, precision, exponent, polynomial_precision=[0, 0], precision_correction=False):
        super(TypicalErrorModel, self).__init__(input_distribution, precision, exponent, polynomial_precision)
        self.name = "TE_" + input_distribution.getName()
        self.precision_correction = precision_correction
        if self.precision is None:
            if self.precision_correction:
                raise ValueError("Specify a precision to use precision_correction!")
        else:
            if self.polynomial_precision is None:
                self.polynomial_precision = [floor(500.0)/self.precision, floor(80.0)/self.precision]


    def _left_segment(self, x):
        if self.precision_correction and self.precision is not None:
            p = self.precision - 1
            u = 2 ** (-p - 1)
            alpha = np.floor((2 ** p) * (-1 / x - 1) + 0.5)
            y = 1 / ((2 ** p) * (1 - u * x) * (1 - u * x)) * (
                    2.0 / 3.0 + 0.5 * alpha + (2 ** (-p - 2) * alpha * (alpha - 1)))
        else:
            y = 0.5 * ((1.0 / x) - 1.0) + 0.25 * (((1.0 / x) - 1.0) ** 2)
        return y

    def _middle_segment(self, x):
        if self.precision_correction:
            p = self.precision - 1
            u = 2 ** (-p - 1)
            y = 1 / ((2 ** p) * (1 - u * x) * (1 - u * x)) * (2.0 / 3.0 + 3 * ((2 ** p) - 1) / 4.0)
        else:
            y = 0.75
        return y

    def _right_segment(self, x):
        if self.precision_correction:
            p = self.precision - 1
            u = 2 ** (-p - 1)
            alpha = np.floor((2 ** p) * (1 / x - 1) - 0.5)
            y = 1 / ((2 ** p) * (1 - u * x) * (1 - u * x)) * (
                    2.0 / 3.0 + (0.5 * alpha) + ((2 ** (-p - 2)) * alpha * (alpha - 1)))
        else:
            y = 0.5 * ((1.0 / x) - 1.0) + 0.25 * (((1.0 / x) - 1.0) ** 2)
        return y


