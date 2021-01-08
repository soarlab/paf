from error_model import HighPrecisionErrorModel, LowPrecisionErrorModel, TypicalErrorModel, ErrorModelWrapper
from cdf_op_dev import ApproximatingPair, IndependentOperation
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


def test_Approx_Operations():
    n = 100
    epsilon = 0.0
    left_dist = UniformDistr(1, 2)
    right_dist = UniformDistr(-1, 1)
    left_operand = ApproximatingPair(n, left_dist)
    right_operand = ApproximatingPair(n, right_dist)
    operation = "/"
    left_exact = []
    right_exact = []
    operation_exact = []
    op = IndependentOperation(operation, left_operand, right_operand)
    op.perform_operation()
    Z = left_dist / right_dist
    for i in range(0, left_operand.n):
        left_exact.append(left_dist.cdf(left_operand.range_array[i]))
        right_exact.append(right_dist.cdf(right_operand.range_array[i]))
        operation_exact.append(Z.cdf(op.output.range_array[i]))
    plt.close("all")
    matplotlib.rcParams.update({'font.size': 10})
    fig, a = plt.subplots(3)
    a[0].plot(left_operand.range_array, left_operand.lower_array, "r", drawstyle='steps-post')
    a[0].plot(left_operand.range_array, left_operand.upper_array, "g", drawstyle='steps-pre')
    a[0].plot(left_operand.range_array, left_exact, "b")
    a[0].set_title("Left operand:"+left_dist.getName()+" (epsilon="+str(epsilon)+")")
    a[1].plot(right_operand.range_array, right_operand.lower_array, "r", drawstyle='steps-post')
    a[1].plot(right_operand.range_array, right_operand.upper_array, "g", drawstyle='steps-pre')
    a[1].plot(right_operand.range_array, right_exact, "b")
    a[1].set_title("Right operand:"+right_dist.getName()+" (epsilon="+str(epsilon)+")")
    a[2].plot(op.output.range_array, op.output.lower_array, "r", drawstyle='steps-post')
    a[2].plot(op.output.range_array, op.output.upper_array, "g", drawstyle='steps-pre')
    a[2].plot(op.output.range_array, operation_exact, "b")
    a[2].set_title("Operation:"+left_dist.getName()+operation+right_dist.getName())
    plt.show()
    #plt.savefig("tests/"+left_dist.getName()+operation+right_dist.getName())
    #plt.close("all")
    exit(0)

def test_Mixed_Operations_Right():
    n = 20
    epsilon = 0.0
    left_dist = UniformDistr(1, 2)
    right_dist = UniformDistr(1, 2)
    left_operand = ApproximatingPair(n, left_dist)
    right_operand = right_dist
    operation = "*"
    left_exact = []
    right_exact = []
    right_range = []
    operation_exact = []
    op = IndependentOperation(operation, left_operand, right_operand)
    op.perform_operation()
    Z = left_dist * right_dist
    for i in range(0, left_operand.n):
        right_range.append(right_operand.range_()[0] + (i / n) * (right_operand.range_()[1] - right_operand.range_()[0]))
        left_exact.append(left_dist.cdf(left_operand.range_array[i]))
        right_exact.append(right_dist.cdf(right_range[i]))
        operation_exact.append(Z.cdf(op.output.range_array[i]))
    plt.close("all")
    matplotlib.rcParams.update({'font.size': 10})
    fig, a = plt.subplots(3)
    a[0].plot(left_operand.range_array, left_operand.lower_array, "r", drawstyle='steps-post')
    a[0].plot(left_operand.range_array, left_operand.upper_array, "g", drawstyle='steps-pre')
    a[0].plot(left_operand.range_array, left_exact, "b")
    a[0].set_title("Left operand:" + left_dist.getName() + " (epsilon=" + str(epsilon) + ")")
    a[1].plot(right_range, right_exact, "b")
    a[1].set_title("Right operand:" + right_dist.getName() + " (epsilon=" + str(epsilon) + ")")
    a[2].plot(op.output.range_array, op.output.lower_array, "r", drawstyle='steps-post')
    a[2].plot(op.output.range_array, op.output.upper_array, "g", drawstyle='steps-pre')
    a[2].plot(op.output.range_array, operation_exact, "b")
    a[2].set_title("Operation:" + left_dist.getName() + operation + right_dist.getName())
    plt.show()
    # plt.savefig("tests/"+left_dist.getName()+operation+right_dist.getName())
    # plt.close("all")
    exit(0)

def test_Mixed_Operations_Left():
    n = 20
    epsilon = 0.0
    left_dist = UniformDistr(-1, 3)
    left_operand = left_dist
    right_dist = UniformDistr(-1, 2)
    right_operand = ApproximatingPair(n, right_dist)
    operation = "*"
    left_exact = []
    right_exact = []
    left_range = []
    operation_exact = []
    op = IndependentOperation(operation, left_operand, right_operand)
    op.perform_operation()
    Z = left_dist * right_dist
    for i in range(0, right_operand.n):
        left_range.append(left_operand.range_()[0] + (i / n) * (left_operand.range_()[1] - left_operand.range_()[0]))
        right_exact.append(right_dist.cdf(right_operand.range_array[i]))
        left_exact.append(left_dist.cdf(left_range[i]))
        operation_exact.append(Z.cdf(op.output.range_array[i]))
    plt.close("all")
    matplotlib.rcParams.update({'font.size': 10})
    fig, a = plt.subplots(3)
    a[0].plot(left_range, left_exact, "b")
    a[0].set_title("Left operand:" + left_dist.getName() + " (epsilon=" + str(epsilon) + ")")
    a[1].plot(right_operand.range_array, right_operand.lower_array, "r", drawstyle='steps-post')
    a[1].plot(right_operand.range_array, right_operand.upper_array, "g", drawstyle='steps-pre')
    a[1].plot(right_operand.range_array, right_exact, "b")
    a[1].set_title("Right operand:" + right_dist.getName() + " (epsilon=" + str(epsilon) + ")")
    a[2].plot(op.output.range_array, op.output.lower_array, "r", drawstyle='steps-post')
    a[2].plot(op.output.range_array, op.output.upper_array, "g", drawstyle='steps-pre')
    a[2].plot(op.output.range_array, operation_exact, "b")
    a[2].set_title("Operation:" + left_dist.getName() + operation + right_dist.getName())
    plt.show()
    # plt.savefig("tests/"+left_dist.getName()+operation+right_dist.getName())
    # plt.close("all")
    exit(0)

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
