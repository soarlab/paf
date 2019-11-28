from model import *
import math
from error_model import *
import numpy as np
import matplotlib.pyplot as plt
from regularizer import *

plt.rcParams.update({'font.size': 30})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'legend.frameon': False})
plt.rcParams.update({'legend.handletextpad': 0.1})
plt.rcParams.update({'legend.labelspacing': 0.5})
plt.rcParams.update({'axes.labelpad': 20})
plt.rcParams.update({'legend.loc':'upper right'})

def copy_tree(my_tree):
    if my_tree.leaf:
        copied_tree = BinaryTree(my_tree.value.name, my_tree.value)
    else:
        copied_tree = BinaryTree(my_tree.value.operator, None, copy_tree(my_tree.children[0]),
                                 copy_tree(my_tree.children[1]))
    return copied_tree


class BinaryTree(object):

    def __init__(self, name, value, left=None, right=None):
        self.root_name = name
        self.root_value = value
        self.left = left
        self.right = right

class Triple:
    def __init__(self, distribution, errorTerm, eps):
        self.distribution=distribution
        self.errorTerm=errorTerm
        self.eps=eps
        self.quantizationTerm=(1.0+(eps*errorTerm))
        self.quantizationTerm.init_piecewise_pdf()

    def compute(self, singleValue):
        tmp=singleValue*self.quantizationTerm


def isPointMassDistr(dist):
    if dist.distribution.range_()[0] == dist.distribution.range_()[-1]:
        return True
    return False

class TreeModel:

    def __init__(self, my_yacc, precision, exp, poly_precision, initialize=True):
        self.initialize = initialize
        self.precision = precision
        self.exp = exp
        self.poly_precision = poly_precision
        # Copy structure of the tree from my_yacc
        self.tree = copy_tree(my_yacc.expression)
        # Evaluate tree
        self.eps = 2 ** (-self.precision)
        self.evaluate(self.tree)

    def evaluate(self, tree):
        """ Recursively populate the Tree with the triples
        (distribution, error distribution, quantized distribution) """
        triple = []
        # Test if we're at a leaf
        if tree.root_value is not None:
            # Non-quantized distribution
            dist = UnOpDist(tree.root_value.distribution)
            # initialize=True means we quantize the inputs
            if self.initialize:
                # Compute error model
                if isPointMassDistr(dist):
                    error = ErrorModelPointMass(dist, self.precision, self.exp)
                    quantized_distribution = quantizedPointMass(dist,self.precision, self.exp)
                else:
                    error = ErrorModel(dist, self.precision, self.exp, self.poly_precision)
                    quantized_distribution = BinOpDist(dist.execute(), "*+", (self.eps*error.distribution))

            # Else we leave the leaf distribution unchanged
            else:
                error = 0
                quantized_distribution = dist
        # If not at a leaf we need to get the distribution and quantized distributions of the children nodes.
        # Then, check the operation. For each operation the template is the same:
        # dist will be the non-quantized operation the non-quantized children nodes
        # qdist will be the non-quantized operation on the quantized children nodes
        # quantized_distribution will be the quantized operation on the quantized children nodes
        else:
            self.evaluate(tree.left)
            self.evaluate(tree.right)
            dist = BinOpDist(tree.left.root_value[0].execute(), tree.root_name, tree.right.root_value[0].execute())
            qdist = BinOpDist(tree.left.root_value[2].execute(), tree.root_name, tree.right.root_value[2].execute())
            error = ErrorModel(qdist, self.precision, self.exp, self.poly_precision)
            quantized_distribution = BinOpDist(qdist.execute(), "*+", (self.eps*error.distribution))
        # We now populate the triple with distribution, error model, quantized distribution '''
        triple.append(dist)
        triple.append(error)
        triple.append(quantized_distribution)
        tree.root_value = triple

    def generate_output_samples(self, sample_nb):
        """ Generate sample_nb samples of tree evaluation in the tree's working precision
            :return an array of samples """
        d = np.zeros(sample_nb)
        setCurrentContextPrecision(self.precision, self.exp)
        for i in range(0, sample_nb):
            d[i] = float(printMPFRExactly(self.evaluate_at_sample(self.tree)))
        resetContextDefault()
        return d

    def evaluate_at_sample(self, tree):
        """ Sample from the leaf then evaluate tree in the tree's working precision"""
        if tree.left is not None or tree.right is not None:
           if tree.left is not None:
               sample_l = self.evaluate_at_sample(tree.left)
           if tree.right is not None:
               sample_r = self.evaluate_at_sample(tree.right)
           if tree.root_name == "+":
               return gmpy2.add(mpfr(str(sample_l)), mpfr(str(sample_r)))
           elif tree.root_name == "-":
               return gmpy2.sub(mpfr(str(sample_l)), mpfr(str(sample_r)))
           elif tree.root_name == "*":
               return gmpy2.mul(mpfr(str(sample_l)), mpfr(str(sample_r)))
           elif tree.root_name == "/":
               return gmpy2.div(mpfr(str(sample_l)), mpfr(str(sample_r)))
           else:
               print("Operation not supported!")
               exit(-1)
        else:
           sample = tree.root_value[0].execute().rand()
           return sample

    def generate_error_samples(self, sample_nb):
        """ Generate sample_nb samples of tree evaluation in the tree's working precision
                    :return an array of samples """
        e = np.zeros(sample_nb)
        setCurrentContextPrecision(self.precision, self.exp)
        for i in range(0, sample_nb):
            sample, lp_sample = self.evaluate_error_at_sample(self.tree)
            e[i] = (sample - lp_sample) / (self.eps * sample)
        resetContextDefault()
        return e

    def evaluate_error_at_sample(self, tree):
        """ Sample from the leaf then evaluate tree in the tree's working precision"""
        if tree.left is not None or tree.right is not None:
            if tree.left is not None:
                sample_l, lp_sample_l = self.evaluate_error_at_sample(tree.left)
            if tree.right is not None:
                sample_r, lp_sample_r = self.evaluate_error_at_sample(tree.right)
            if tree.root_name == "+":
                return (sample_l + sample_r), gmpy2.add(mpfr(str(lp_sample_l)), mpfr(str(lp_sample_r)))
            elif tree.root_name == "-":
                return (sample_l - sample_r), gmpy2.sub(mpfr(str(lp_sample_l)), mpfr(str(lp_sample_r)))
            elif tree.root_name == "*":
                return (sample_l * sample_r), gmpy2.mul(mpfr(str(lp_sample_l)), mpfr(str(lp_sample_r)))
            elif tree.root_name == "/":
                return (sample_l / sample_r), gmpy2.div(mpfr(str(lp_sample_l)), mpfr(str(lp_sample_r)))
            else:
                print("Operation not supported!")
                exit(-1)
        else:
            sample = tree.root_value[0].execute().rand()
            return sample, mpfr(str(sample))

    def plot_range_analysis(self, sample_nb, file_name):
        r = self.generate_output_samples(sample_nb)
        a = self.tree.root_value[2].a
        b = self.tree.root_value[2].b
        # as bins, choose all the intervals between successive pairs of representable numbers between a and b
        bins = []
        setCurrentContextPrecision(self.precision, self.exp)
        f = mpfr(str(a))
        if a < f:
            f = gmpy2.next_below(f)
        while f < b:
            bins.append(float(printMPFRExactly(f)))
            f = gmpy2.next_above(f)
        resetContextDefault()
        plt.figure(file_name, figsize=(15,10))
        plt.hist(r, bins, density=True, color="b")
        x = np.linspace(a, b, 1000)
        plt.plot(x, self.tree.root_value[2].distribution.get_piecewise_pdf()(x), linewidth=7, color="red")
        plotTicks(file_name, ticks=[7.979, 16.031], label="FPT: [7.979, 16.031]")
        plotBoundsDistr(file_name, self.tree.root_value[2].distribution)
        plt.xlabel('Distribution Range')
        plt.ylabel('PDF')
        plt.legend()
        plt.savefig("./pics/"+file_name, dpi = 100)
        plt.close("all")

    def plot_empirical_error_distribution(self, sample_nb, file_name):
        e = self.generate_error_samples(sample_nb)
        a = math.floor(e.min())
        b = math.ceil(e.max())
        # as bins, choose multiples of 2*eps between a and b
        bins = np.linspace(a, b, (b-a) * 2**(self.precision-1))
        plt.hist(e, bins, density=True)
        plt.savefig("pics/" + file_name)
        plt.close("all")

class quantizedPointMass:

   def __init__(self, wrapperInputDistribution, precision, exp):
       self.wrapperInputDistribution = wrapperInputDistribution
       self.inputdistribution = self.wrapperInputDistribution.execute()
       self.precision = precision
       self.exp = exp
       setCurrentContextPrecision(self.precision, self.exp)
       qValue = printMPFRExactly(mpfr(str(self.inputdistribution)))
       resetContextDefault()
       self.distribution = ConstDistr(float(qValue))
       self.distribution.init_piecewise_pdf()

   def execute(self):
       return self.distribution

class BinOpDist:
    """
    Wrapper class for the result of an arithmetic operation on PaCal distributions
    Warning! leftoperand and rightoperant MUST be PaCal distributions
    """
    def __init__(self, leftoperand, operator, rightoperand, regularize=True):
        self.leftoperand = leftoperand
        self.operator = operator
        self.rightoperand = rightoperand

        if operator == "+":
            self.distribution = self.leftoperand + self.rightoperand
        elif operator == "-":
            self.distribution = self.leftoperand - self.rightoperand
        elif operator == "*":
            self.distribution = self.leftoperand * self.rightoperand
        elif operator == "/":
            self.distribution = self.leftoperand / self.rightoperand
        # operator to multiply by a relative error
        elif operator == "*+":
            self.distribution = self.leftoperand * (1 + self.rightoperand)
        else:
            print("Operation not supported!")
            exit(-1)

        self.distribution.init_piecewise_pdf()

        if regularize:
            self.distribution = chebfunInterpDistr(self.distribution, 10)
        #self.distribution=chebfunInterpDistr(self.distribution,5)

        self.a = self.distribution.range_()[0]
        self.b = self.distribution.range_()[-1]

    def execute(self):
        return self.distribution


class UnOpDist:
    """
    Wrapper class for the result of unary operation on a PaCal distribution
    """

    def __init__(self, operand, operation=None):
        if operation is None:
            self.distribution = operand
            self.a = self.distribution.range_()[0]
            self.b = self.distribution.range_()[-1]
        else:
            print("Unary operation not yet supported")
            exit(-1)

    def execute(self):
        return self.distribution
