from time import time
from pacal import UniformDistr
import matplotlib.pyplot as plt

from abstract_error_model import AbstractErrorModel
import numpy as np

###
# Approximate Error Model given by the "Typical Distribution"
###
from wrapper_error_model import ErrorModelWrapper


class TypicalErrorModel(AbstractErrorModel):
    """
    An implementation of the typical error distribution with three segments
    """
    def __init__(self, input_distribution=None, precision=None, **kwargs):
        super(TypicalErrorModel, self).__init__(input_distribution, precision, None)
        self.name = "TypicalErrorDistribution"
        if precision is not None:
            self.p = precision-1
        else:
            self.p = None

    def _left_segment(self, x):
        if self.p is None:
            y = 0.5 * ((1.0 / x) - 1.0) + 0.25 * (((1.0 / x) - 1.0) ** 2)
        else:
            u = 2 ** (-self.p - 1)
            alpha = np.floor(2 ** self.p * (-1 / x - 1) + 0.5)
            y = 1 / (2 ** self.p * (1 - u * x) ** 2) * (
                    2 / 3 + 0.5 * alpha + 2 ** (-self.p - 2) * alpha * (alpha - 1))
        return y

    def _middle_segment(self, x):
        if self.p is None:
            y = 0.75
        else:
            u = 2 ** (-self.p - 1)
            #2/3 is an integer operation in python2. I know in python3 this is not the
            # case, but it is really dangerous to do such thing. Please use 2.0/3.0.
            y = 1 / (2 ** self.p * (1 - u * x) ** 2) * (2 / 3 + 3 * (2 ** self.p - 1) / 4)
        return y

    def _right_segment(self, x):
        if self.p is None:
            y = 0.5 * ((1.0 / x) - 1.0) + 0.25 * (((1.0 / x) - 1.0) ** 2)
        else:
            u = 2 ** (-self.p - 1)
            #This is really dangerous, please use parenthesis.
            alpha = np.floor(2 ** self.p * (1 / x - 1) - 0.5)
            y = 1 / (2 ** self.p * (1 - u * x) ** 2) * (
                    2 / 3 + 0.5 * alpha + 2 ** (-self.p - 2) * alpha * (alpha - 1))
        return y


###
# TESTS
###

def test_typical_error_model():
    t = time()
    E = TypicalErrorModel()
    E.init_piecewise_pdf()
    print(E.getName())
    print(E.int_error())
    # Test comparing with nothing. Should return the string N/A
    print(E.compare())
    print(time() - t)
    U = UniformDistr(4, 32)
    E = TypicalErrorModel(U)
    E.init_piecewise_pdf()
    print(E.getName())
    print(E.int_error())
    # Test comparing with U, precision unspecified
    print(E.compare())
    print(time() - t)
    E = TypicalErrorModel(U, 9)
    E.init_piecewise_pdf()
    print(E.getName())
    print(E.int_error())
    # Test comparing with U, precision specified
    print(E.compare())
    print(time() - t)
    # A distribution badly approximated by the TypicalErrorModel
    U = UniformDistr(4, 6)
    E = TypicalErrorModel(U)
    E.init_piecewise_pdf()
    print(E.getName())
    print(E.int_error())
    # Comparison should show poor KS-statistics
    print(E.compare())
    print(time() - t)
    # test wrapper
    wrapper = ErrorModelWrapper(E)
    print(wrapper.getName())
    s = wrapper.getSampleSet()
    plt.hist(s, range=[-1, 1], density=True)
