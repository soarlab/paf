from error_model import HighPrecisionErrorModel, LowPrecisionErrorModel, TypicalErrorModel, ErrorModelWrapper
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import ntpath
from time import time
from pacal import UniformDistr, NormalDistr, BetaDistr
from fpryacc import FPRyacc
from tree_model import TreeModel

###
# TreeModel Tests
###

def test_TreeModel():
    # typical model test
    exponent = 8
    mantissa = 24
    file = "tests/my_expression.txt"
    f = open(file, "r")
    file_name = (ntpath.basename(file).split(".")[0]).lower()  # (file.split(".")[0]).lower()
    text = f.read()
    text = text[:-1]
    f.close()
    myYacc = FPRyacc(text, False)
    start_time = time()
    T = TreeModel(myYacc, mantissa, exponent, [40, 10], 100, 250000, error_model="high_precision")
    end_time = time()
    print("Exe time --- %s seconds ---" % (end_time - start_time))
    plt.close("all")
    matplotlib.rcParams.update({'font.size': 11})
    fig, a = plt.subplots(2, 2)
    dist = T.tree.root_value[0].distribution
    t0 = dist.range_()[0]
    tf = dist.range_()[1]
    x = np.linspace(t0, tf, 100)
    a[0][0].plot(x, dist.pdf(x))
    a[0][0].set_title("Unquantized operations")
    dist = T.tree.root_value[1].distribution
    t0 = dist.range_()[0]
    tf = dist.range_()[1]
    x = np.linspace(t0, tf, 100)
    a[0][1].plot(x, dist.pdf(x))
    a[0][1].set_title("Last error")
    dist = T.tree.root_value[2].distribution
    t0 = dist.range_()[0]
    tf = dist.range_()[1]
    x = np.linspace(t0, tf, 100)
    a[1][0].plot(x, dist.pdf(x))
    a[1][0].set_title("Quantized operations")
    plt.show()

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
    E = TypicalErrorModel(U, precision_correction=True)
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
