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
    precision: if specified a correction will be applied. Only makes a notable difference at very low precision (< 7 bits)
    polynomial_precision: should be left unspecified when the precision is unspecified, the typical distribution is so
                          regular that pychebfun always converges. This is not the case when the precision correction
                          is applied, so default is to then assign values dynamically
    """
    def __init__(self, input_distribution=None, precision=None, exponent=None, polynomial_precision=[0, 0]):
        super(TypicalErrorModel, self).__init__(input_distribution, precision, exponent, polynomial_precision)
        self.name = "TypicalErrorDistribution"
        if self.precision is not None:
            self.polynomial_precision = [floor(500.0)/self.precision, floor(80.0)/self.precision]

    def _left_segment(self, x):
        if self.precision is None:
            y = 0.5 * ((1.0 / x) - 1.0) + 0.25 * (((1.0 / x) - 1.0) ** 2)
        else:
            p = self.precision - 1
            u = 2 ** (-p - 1)
            alpha = np.floor((2 ** p) * (-1 / x - 1) + 0.5)
            y = 1 / ((2 ** p) * (1 - u * x) * (1 - u * x)) * (
                    2.0 / 3.0 + 0.5 * alpha + (2 ** (-p - 2) * alpha * (alpha - 1)))
        return y

    def _middle_segment(self, x):
        if self.precision is None:
            y = 0.75
        else:
            p = self.precision - 1
            u = 2 ** (-p - 1)
            y = 1 / ((2 ** p) * (1 - u * x) * (1 - u * x)) * (2.0 / 3.0 + 3 * ((2 ** p) - 1) / 4.0)
        return y

    def _right_segment(self, x):
        if self.precision is None:
            y = 0.5 * ((1.0 / x) - 1.0) + 0.25 * (((1.0 / x) - 1.0) ** 2)
        else:
            p = self.precision - 1
            u = 2 ** (-p - 1)
            alpha = np.floor((2 ** p) * (1 / x - 1) - 0.5)
            y = 1 / ((2 ** p) * (1 - u * x) * (1 - u * x)) * (
                    2.0 / 3.0 + (0.5 * alpha) + ((2 ** (-p - 2)) * alpha * (alpha - 1)))
        return y


