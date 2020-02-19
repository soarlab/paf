import numpy as np
import gmpy2
from gmpy2 import mpfr
import time
import os

from abstract_error_model import AbstractErrorModel
from point_mass_error_model import ErrorModelPointMass
from typical_error_model import TypicalErrorModel
from wrapper_error_model import ErrorModelWrapper
from model import UnaryOperation
from operations import quantizedPointMass, BinOpDist, UnOpDist
from setup_utils import loadIfExists, storage_path
from project_utils import printMPFRExactly, resetContextDefault, setCurrentContextPrecision


def copy_tree(my_tree):
    if my_tree.leaf:
        copied_tree = BinaryTree(my_tree.value.name, my_tree.value)
    else:
        if isinstance(my_tree.value, UnaryOperation):
            copied_tree = UnaryTree(my_tree.value.operator, None, copy_tree(my_tree.children[0]))
        else:
            copied_tree = BinaryTree(my_tree.value.operator, None,
                                     copy_tree(my_tree.children[0]),
                                     copy_tree(my_tree.children[1]), my_tree.value.indipendent)
    return copied_tree


class BinaryTree(object):
    def __init__(self, name, value, left=None, right=None, convolution=True):
        self.root_name = name
        self.root_value = value
        self.left = left
        self.right = right
        self.convolution = convolution

class UnaryTree(object):
    def __init__(self, name, value, onlychild):
        self.root_name = name
        self.root_value = value
        self.left = onlychild
        self.right = None

def isPointMassDistr(dist):
    if dist.distribution.range_()[0] == dist.distribution.range_()[-1]:
        return True
    return False

class DistributionsManager:
    def __init__(self, samples_dep_op):
        self.samples_dep_op = samples_dep_op
        self.errordictionary = {}
        self.distrdictionary = {}

    def createErrorModel(self, wrapDist, precision, exp, pol_prec, typical=True):
        if typical:
            if wrapDist.name in self.errordictionary:
                return self.errordictionary[wrapDist.name]
            else:
                tmp=ErrorModelWrapper(TypicalErrorModel(wrapDist), precision, exp)
                #tmp=WrappedHighPrecisionError(wrapDist, precision, exp)
                #tmp=HighPrecisionErrorModel(wrapDist,precision,exp)
                self.errordictionary[wrapDist.name] = tmp
                return tmp
        else:
            if wrapDist.name in self.errordictionary:
                return self.errordictionary[wrapDist.name]
            else:
                tmp=AbstractErrorModel(wrapDist, precision, exp, pol_prec)
                self.errordictionary[wrapDist.name]=tmp
                return tmp

    def createBinOperation(self, leftoperand, operator, rightoperand, poly_precision,  regularize=True, convolution=True):
        name="("+leftoperand.name+str(operator)+rightoperand.name+")"
        if name in self.distrdictionary:
            return self.distrdictionary[name]
        else:
            tmp=BinOpDist(leftoperand, operator, rightoperand, poly_precision, self.samples_dep_op, regularize, convolution)
            self.distrdictionary[name]=tmp
            return tmp

    def createUnaryOperation(self, operand, name, operation=None):
        if operation is not None:
            tmp_name=name+"("+operand.name+")"
        else:
            tmp_name=name
        if tmp_name in self.distrdictionary:
            return  self.distrdictionary[tmp_name]
        else:
            tmp=UnOpDist(operand, tmp_name, operation)
            self.distrdictionary[tmp_name]=tmp
            return tmp

class TreeModel:

    def __init__(self, my_yacc, precision, exp, poly_precision, samples_dep_op, initialize=True):
        self.initialize = initialize
        self.precision = precision
        self.exp = exp
        self.poly_precision = poly_precision
        # Copy structure of the tree from my_yacc
        self.tree = copy_tree(my_yacc.expression)
        # Evaluate tree
        self.eps = 2 ** (-self.precision)
        self.samples_dep_op=samples_dep_op
        self.manager=DistributionsManager(self.samples_dep_op)
        self.evaluate(self.tree)
        self.final_quantized_distr=self.tree.root_value[2]
        self.final_exact_distr=self.tree.root_value[0]
        self.abs_err_distr= UnOpDist(BinOpDist(self.final_quantized_distr, "-", self.final_exact_distr, 1000, self.samples_dep_op,
                                               regularize=True, convolution=False), "abs_err", "abs")

    def evaluate(self, tree):
        """ Recursively populate the Tree with the triples
        (distribution, error distribution, quantized distribution) """
        # Test if we're at a leaf
        if tree.root_value is not None:
            # Non-quantized distribution
            dist = self.manager.createUnaryOperation(tree.root_value, tree.root_name)
            # initialize=True means we quantize the inputs
            if self.initialize:
                # Compute error model
                if isPointMassDistr(dist):
                    error = ErrorModelPointMass(dist, self.precision, self.exp)
                    quantized_distribution = quantizedPointMass(dist,self.precision, self.exp)
                else:
                    error = self.manager.createErrorModel(dist, self.precision, self.exp, self.poly_precision)
                    quantized_distribution = self.manager.createBinOperation(dist, "*+", error, self.poly_precision)
            # Else we leave the leaf distribution unchanged
            else:
                error = 0
                quantized_distribution = dist

        # If not at a leaf we need to get the distribution and quantized distributions of the children nodes.
        # Then, check the operation. For each operation the template is the same:
        # dist will be the non-quantized operation the non-quantized children nodes
        # qdist will be the non-quantized operation on the quantized children nodes
        # quantized_distribution will be the quantized operation on the quantized children nodes

        elif tree.left is not None and tree.right is not None:

            self.evaluate(tree.left)
            self.evaluate(tree.right)

            dist  = self.manager.createBinOperation(tree.left.root_value[0], tree.root_name, tree.right.root_value[0], self.poly_precision, convolution=tree.convolution)
            qdist = self.manager.createBinOperation(tree.left.root_value[2], tree.root_name, tree.right.root_value[2], self.poly_precision, convolution=tree.convolution)

            if isPointMassDistr(dist):
                error = ErrorModelPointMass(qdist, self.precision, self.exp)
                quantized_distribution = quantizedPointMass(dist, self.precision, self.exp)

            else:
                error = self.manager.createErrorModel(qdist, self.precision, self.exp, self.poly_precision)
                quantized_distribution = self.manager.createBinOperation(qdist, "*+", error, self.poly_precision)
        else:
            self.evaluate(tree.left)
            dist = self.manager.createUnaryOperation(tree.left.root_value[0], tree.root_name, tree.root_name)
            qdist = self.manager.createUnaryOperation(tree.left.root_value[2], tree.root_name, tree.root_name)

            if isPointMassDistr(dist):
                error = ErrorModelPointMass(qdist, self.precision, self.exp)
                quantized_distribution = quantizedPointMass(dist, self.precision, self.exp)
            else:
                error = self.manager.createErrorModel(qdist, self.precision, self.exp, self.poly_precision)
                quantized_distribution = self.manager.createBinOperation(qdist, "*+", error, self.poly_precision)

        # We now populate the triple with distribution, error model, quantized distribution '''
        tree.root_value = [dist, error, quantized_distribution]

    def resetInit(self, tree):
        if tree.left is not None or tree.right is not None:
            if tree.left is not None:
                self.resetInit(tree.left)
            if tree.right is not None:
                self.resetInit(tree.right)
            tree.root_value[0].resetSampleInit()
        else:
            tree.root_value[0].resetSampleInit()

    def generate_error_samples(self, sample_time, name, golden=False):
        """ Generate sample_nb samples of tree evaluation in the tree's working precision
                    :return an array of samples, absolute errors and relative errors """

        if golden and loadIfExists and os.path.exists(storage_path + name):
            print("Golden distribution is going to be loaded from disk!")
            return True, [], [], []

        print("Generating Samples...")

        rel_err = []
        abs_err = []
        values = []
        setCurrentContextPrecision(self.precision, self.exp)
        start_time = time.time()
        end_time = 0
        while end_time <= sample_time:
            self.resetInit(self.tree)
            sample, lp_sample = self.evaluate_error_at_sample(self.tree)
            values.append(sample)
            tmp_abs = abs(float(printMPFRExactly(lp_sample)) - sample)
            abs_err.append(tmp_abs)
            rel_err.append(tmp_abs / sample)
            end_time = time.time() - start_time
        self.resetInit(self.tree)
        resetContextDefault()
        print("... Done with generation")
        return False, np.asarray(values), np.asarray(abs_err), np.asarray(rel_err)

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
            elif tree.root_name == "exp":
                return np.exp(sample_l), gmpy2.exp(mpfr(str(sample_l)))
            elif tree.root_name == "sin":
                return np.sin(sample_l), gmpy2.sin(mpfr(str(sample_l)))
            elif tree.root_name == "cos":
                return np.cos(sample_l), gmpy2.cos(mpfr(str(sample_l)))
            else:
                print("Operation not supported!")
                exit(-1)
        else:
            sample = tree.root_value[0].getSampleSet(n=1)[0]
            return sample, mpfr(str(sample))