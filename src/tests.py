from error_model import HighPrecisionErrorModel, LowPrecisionErrorModel, TypicalErrorModel, ErrorModelWrapper
import matplotlib.pyplot as plt
from time import time
from pacal import UniformDistr, NormalDistr, BetaDistr

###
# Error Models Tests
###

def test_HP_error_model():
    exponent = 8
    mantissa = 24
    t = time()
    poly_precision = [50, 20]
    U = UniformDistr(4, 32)
    E = HighPrecisionErrorModel(U, mantissa, exponent, poly_precision)
    E.init_piecewise_pdf()
    print(E.getName())
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    t = time()
    U = UniformDistr(4, 5)
    E = HighPrecisionErrorModel(U, mantissa, exponent, poly_precision)
    E.init_piecewise_pdf()
    print(E.getName())
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    t = time()
    U = UniformDistr(7, 8)
    E = HighPrecisionErrorModel(U, mantissa, exponent, poly_precision)
    E.init_piecewise_pdf()
    print(E.getName())
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    t = time()
    U = NormalDistr()
    E = HighPrecisionErrorModel(U, mantissa, exponent, poly_precision)
    E.init_piecewise_pdf()
    print(E.getName())
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    # test wrapper
    wrapper = ErrorModelWrapper(E)
    print(wrapper.getName())
    s = wrapper.getSampleSet()
    plt.hist(s, range=[-1, 1], density=True)


def test_LP_error_model():
    exponent = 3
    mantissa = 4
    poly_precision = [0, 0]
    t = time()
    D = UniformDistr(-1, 1)
    E = LowPrecisionErrorModel(D, mantissa, exponent, poly_precision)
    print(E.int_error())
    print(E.compare(100000))
    print(time() - t)
    t = time()
    D = NormalDistr()
    E = LowPrecisionErrorModel(D, mantissa, exponent, poly_precision)
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    t = time()
    D = BetaDistr(2, 2)
    E = LowPrecisionErrorModel(D, mantissa, exponent, poly_precision)
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    exponent = 5
    mantissa = 11
    t = time()
    D = UniformDistr(2, 4)
    E = LowPrecisionErrorModel(D, mantissa, exponent, poly_precision)
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    t = time()
    D = NormalDistr()
    E = LowPrecisionErrorModel(D, mantissa, exponent, poly_precision)
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    t = time()
    D = BetaDistr(2, 2)
    E = LowPrecisionErrorModel(D, mantissa, exponent, poly_precision)
    print(E.int_error())
    print(E.compare())
    print(time() - t)
    # test wrapper
    wrapper = ErrorModelWrapper(E)
    print(wrapper.getName())
    s = wrapper.getSampleSet()
    plt.hist(s, range=[-1, 1], density=True)


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
    E = TypicalErrorModel(U, 8)
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
