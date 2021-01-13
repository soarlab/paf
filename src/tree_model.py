import numpy as np
import gmpy2
from gmpy2 import mpfr
import time
import os

import SMT_Interface
from IntervalArithmeticLibrary import Interval
from SymbolicAffineArithmetic import SymbolicToGelpia
from error_model import HighPrecisionErrorModel, LowPrecisionErrorModel, FastTypicalErrorModel, ErrorModelPointMass, \
    ErrorModelWrapper, TypicalErrorModel
from model import UnaryOperation
from operations import quantizedPointMass, BinOpDist, UnOpDist, pacal, plt, ConstantManager
from setup_utils import loadIfExists, storage_path, global_interpolate, constraints_probabilities, digits_for_range
from project_utils import printMPFRExactly, reset_default_precision, set_context_precision, isNumeric


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
    def __init__(self, samples_dep_op, dependent_mode):
        self.samples_dep_op = samples_dep_op
        self.dependent_mode = dependent_mode
        self.errordictionary = {}
        self.distrdictionary = {}

    def createErrorModel(self, wrapDist, precision, exp, pol_prec, interp_precision, error_model):
        if error_model == "typical":
            if wrapDist.name in self.errordictionary:
                return self.errordictionary[wrapDist.name]
            else:
                tmp = ErrorModelWrapper(FastTypicalErrorModel(wrapDist.distribution, wrapDist.name, precision, exp, interp_precision))
                #tmp = ErrorModelWrapper(TypicalErrorModel(precision=precision))
                self.errordictionary[wrapDist.name] = tmp
                return tmp
        elif error_model == "high_precision":
            if wrapDist.name in self.errordictionary:
                return self.errordictionary[wrapDist.name]
            else:
                tmp = ErrorModelWrapper(HighPrecisionErrorModel(wrapDist.distribution, wrapDist.name, precision, exp, pol_prec))
                self.errordictionary[wrapDist.name] = tmp
                return tmp
        elif error_model == "low_precision":
            if wrapDist.name in self.errordictionary:
                return self.errordictionary[wrapDist.name]
            else:
                tmp = ErrorModelWrapper(LowPrecisionErrorModel(wrapDist.distribution, wrapDist.name, precision, exp, pol_prec))
                self.errordictionary[wrapDist.name] = tmp
                return tmp
        else:
            raise ValueError('Invalid ErrorModel name.')

    def createBinOperation(self, leftoperand, operator, rightoperand,
                           interp_precision, exact_affine_forms=None, smt_triple=None, regularize=True, convolution=True):
        name = "(" + leftoperand.name + str(operator) + rightoperand.name + ")"
        if name in self.distrdictionary:
            return self.distrdictionary[name]
        else:
            tmp = BinOpDist(leftoperand, operator, rightoperand, smt_triple, name,
                            interp_precision, self.samples_dep_op, exact_affine_forms,
                            regularize, convolution, self.dependent_mode)
            self.distrdictionary[name] = tmp
            return tmp

    def createUnaryOperation(self, operand, name, operation=None):
        if operation is not None:
            tmp_name = operation + "(" + operand.name + ")"
        else:
            tmp_name = name
        if tmp_name in self.distrdictionary:
            return self.distrdictionary[tmp_name]
        else:
            tmp = UnOpDist(operand, tmp_name, operation)
            self.distrdictionary[tmp_name] = tmp
            return tmp


class TreeModel:
    """
    :param
            poly_precision: the pair [m, n] specifying how the error model constructor interpolates via pychebfun
            interp_precision: integer specifying the interpolation precision in the computation of dependent operations
            error_model: choose between "typical", "high_precision" and "low_precision"
            dependent_mode: specifies how dependent operations are handled, choose between "full_mc" (full Monte Carlo),
                            "hybrid", "analytic" and "auto" (choice is made dynamically at each node)
    """
    def __init__(self, my_yacc, precision, exponent, poly_precision, interp_precision,
                 samples_dep_op, initialize=True, error_model="typical", dependent_mode="p-box"):
        self.initialize = initialize
        self.precision = precision
        self.exponent = exponent
        self.poly_precision = poly_precision
        self.interp_precision = interp_precision
        self.error_model = error_model
        self.dependent_mode = dependent_mode
        self.eps = 2 ** (-self.precision)
        self.samples_dep_op = samples_dep_op
        # Copy structure of the tree from my_yacc
        self.tree = copy_tree(my_yacc.expression)
        # Evaluate tree
        self.manager = DistributionsManager(self.samples_dep_op, dependent_mode)
        self.evaluate(self.tree)

        self.final_quantized_distr = self.tree.root_value[2]
        self.final_exact_distr = self.tree.root_value[0]

        smt_manager_dist=self.tree.root_value[5]
        smt_manager_qdist=self.tree.root_value[6]

        smt_manager=smt_manager_dist.merge_instance(smt_manager_qdist)

        smt_triple = (self.tree.root_value[4], self.tree.root_value[3], smt_manager)

        quantized_interval=self.final_quantized_distr.symbolic_affine.compute_interval()
        print("FP Range Quantized Distribution: "+str(quantized_interval.lower)+", "+str(quantized_interval.upper))
        self.error_results=self.elaborate_Gelpia_error_intervals(self.final_exact_distr.constraints_dict, self.final_quantized_distr.symbolic_error)

        #self.err_distr = BinOpDist(self.final_quantized_distr, "-",
        #                           self.final_exact_distr,
        #                            smt_triple, "err_pbox", 100, self.samples_dep_op,
        #                        regularize=True, convolution=False, dependent_mode="p-box", is_error_computation=True)

        #self.abs_err_distr = UnOpDist(self.err_distr, "abs_err_pbox", "abs")

        #self.lower_error_affine, self.upper_error_affine=self.compute_lower_upper_affine_error()
        #self.lower_error_affine.get_piecewise_cdf()
        #self.upper_error_affine.get_piecewise_cdf()
        #self.relative_err_distr = UnOpDist(BinOpDist(self.abs_err_distr, "/",
        #                                             self.final_exact_distr, 1000, self.samples_dep_op,
        #                                             regularize=True, convolution=False), "rel_err", "abs")

    def evaluate(self, tree):
        """ Recursively populate the Tree with the triples
        (distribution, error distribution, quantized distribution) """
        # Test if we're at a leaf
        if tree.root_value is not None:
            # Non-quantized distribution

            dist = self.manager.createUnaryOperation(tree.root_value, tree.root_name)

            smt_manager_dist=SMT_Interface.SMT_Instance()
            smt_manager_qdist=SMT_Interface.SMT_Instance()

            if smt_manager_dist.check_string_number_is_exp_notation(tree.root_name):
                scientific_name = ConstantManager.get_new_constant_index()
                smt_manager_dist.add_var(scientific_name, dist.discretization.affine.interval.lower,
                                                          dist.discretization.affine.interval.upper)
                smt_manager_qdist.add_var(scientific_name, dist.discretization.affine.interval.lower,
                                                          dist.discretization.affine.interval.upper)
                dist_smt_query = SMT_Interface.create_exp_for_UnaryOperation_SMT_LIB(scientific_name)
            else:
                smt_manager_dist.add_var(tree.root_name, dist.discretization.lower, dist.discretization.upper)
                smt_manager_qdist.add_var(tree.root_name,dist.discretization.lower, dist.discretization.upper)
                dist_smt_query= SMT_Interface.create_exp_for_UnaryOperation_SMT_LIB(tree.root_name)

            # initialize=True means we quantize the inputs
            if self.initialize:
                # Compute error model
                if isPointMassDistr(dist):
                    error = ErrorModelPointMass(dist, self.precision, self.exponent)
                    quantized_distribution = quantizedPointMass(dist, self.precision, self.exponent)
                    quantized_value_name=ConstantManager.get_new_constant_index()
                    smt_manager_qdist.add_var(quantized_value_name, quantized_distribution.discretization.lower, quantized_distribution.discretization.upper)
                    qdist_smt_query = SMT_Interface.create_exp_for_UnaryOperation_SMT_LIB(quantized_value_name)
                else:
                    error = self.manager.createErrorModel(dist, self.precision, self.exponent, self.poly_precision, self.interp_precision,
                                                          self.error_model)
                    exact_affine_forms=[dist.discretization.affine, None,
                                        dist.symbolic_affine, None]
                    quantized_distribution = self.manager.createBinOperation(dist, "*+", error, self.interp_precision, exact_affine_forms=exact_affine_forms)
                    error_name_SMT=SMT_Interface.clean_var_name_SMT(error.distribution.name)
                    smt_manager_qdist.add_var(error_name_SMT, error.discretization.lower, error.discretization.upper)
                    qdist_smt_query = SMT_Interface.create_exp_for_BinaryOperation_SMT_LIB(dist_smt_query, "*+", error_name_SMT)
            # Else we leave the leaf distribution unchanged
            else:
                error = 0
                quantized_distribution = dist
                qdist_smt_query = dist_smt_query

        # If not at a leaf we need to get the distribution and quantized distributions of the children nodes.
        # Then, check the operation. For each operation the template is the same:
        # dist will be the non-quantized operation the non-quantized children nodes
        # qdist will be the non-quantized operation on the quantized children nodes
        # quantized_distribution will be the quantized operation on the quantized children nodes
        # Binary Operation (both left and right children are not None)
        elif tree.left is not None and tree.right is not None:

            self.evaluate(tree.left)
            self.evaluate(tree.right)

            smt_manager_dist=tree.left.root_value[5].merge_instance(tree.right.root_value[5])
            smt_manager_qdist=tree.left.root_value[6].merge_instance(tree.right.root_value[6])


            smt_triple_dist= (tree.left.root_value[3], tree.right.root_value[3], smt_manager_dist)
            smt_triple_qdist = (tree.left.root_value[4], tree.right.root_value[4], smt_manager_qdist)

            exact_affine_forms=[tree.left.root_value[0].discretization.affine,  tree.right.root_value[0].discretization.affine,
                                tree.left.root_value[0].symbolic_affine,        tree.right.root_value[0].symbolic_affine]

            dist = self.manager.createBinOperation(tree.left.root_value[0], tree.root_name, tree.right.root_value[0],
                                                   self.interp_precision, exact_affine_forms, smt_triple_dist, convolution=tree.convolution)

            qdist = self.manager.createBinOperation(tree.left.root_value[2], tree.root_name, tree.right.root_value[2],
                                                    self.interp_precision, exact_affine_forms, smt_triple_qdist, convolution=tree.convolution)

            dist_smt_query= SMT_Interface.create_exp_for_BinaryOperation_SMT_LIB(tree.left.root_value[3], tree.root_name, tree.right.root_value[3])
            qdist_smt_query= SMT_Interface.create_exp_for_BinaryOperation_SMT_LIB(tree.left.root_value[4], tree.root_name, tree.right.root_value[4])


            if isPointMassDistr(dist):
                error = ErrorModelPointMass(qdist, self.precision, self.exponent)
                quantized_distribution = quantizedPointMass(qdist, self.precision, self.exponent)
                qdist_smt_query = SMT_Interface.create_exp_for_UnaryOperation_SMT_LIB(
                    quantized_distribution.getName())
            else:
                error = self.manager.createErrorModel(qdist, self.precision, self.exponent, self.poly_precision,
                                                      self.interp_precision, self.error_model)
                exact_affine_forms = [dist.discretization.affine, None,
                                      dist.symbolic_affine, None]
                quantized_distribution = self.manager.createBinOperation(qdist, "*+", error,
                                                                         self.interp_precision, exact_affine_forms)
                error_name_SMT = SMT_Interface.clean_var_name_SMT(error.distribution.name)
                smt_manager_qdist.add_var(error_name_SMT, error.discretization.lower, error.discretization.upper)
                qdist_smt_query = SMT_Interface.create_exp_for_BinaryOperation_SMT_LIB(qdist_smt_query, "*+",
                                                                                                error_name_SMT)

        # Unary Operation (exp, cos ,sin)
        else:
            self.evaluate(tree.left)
            dist = self.manager.createUnaryOperation(tree.left.root_value[0], tree.root_name, tree.root_name)
            qdist = self.manager.createUnaryOperation(tree.left.root_value[2], tree.root_name, tree.root_name)

            dist_smt_query = SMT_Interface.create_exp_for_UnaryOperation_SMT_LIB(tree.left.root_value[3],
                                                                                 tree.root_name)
            qdist_smt_query = SMT_Interface.create_exp_for_UnaryOperation_SMT_LIB(tree.left.root_value[4],
                                                                                           tree.root_name)
            smt_manager_dist =SMT_Interface.SMT_Instance().merge_instance(tree.left.root_value[5])
            smt_manager_qdist = SMT_Interface.SMT_Instance().merge_instance(tree.left.root_value[6])

            if isPointMassDistr(dist):
                error = ErrorModelPointMass(qdist, self.precision, self.exponent)
                quantized_distribution = quantizedPointMass(qdist, self.precision, self.exponent)
                qdist_smt_query = SMT_Interface.create_exp_for_UnaryOperation_SMT_LIB(
                    quantized_distribution.getName())
            else:
                error = self.manager.createErrorModel(qdist, self.precision, self.exponent, self.poly_precision,
                                                      self.interp_precision, self.error_model)
                exact_affine_forms=[tree.left.root_value[0].discretization.affine, None,
                                    tree.left.root_value[0].symbolic_affine, None]
                quantized_distribution = self.manager.createBinOperation(qdist, "*+", error, self.interp_precision, exact_affine_forms=exact_affine_forms)
                error_name_SMT = SMT_Interface.clean_var_name_SMT(error.distribution.name)
                smt_manager_qdist.add_var(error_name_SMT, error.discretization.lower, error.discretization.upper)
                qdist_smt_query = SMT_Interface.create_exp_for_BinaryOperation_SMT_LIB(qdist_smt_query, "*+",
                                                                                                error_name_SMT)

        # We now populate the triple with distribution, error model, quantized distribution '''
        tree.root_value = [dist, error, quantized_distribution, dist_smt_query, qdist_smt_query, smt_manager_dist, smt_manager_qdist]

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
        err=[]
        values = []
        set_context_precision(self.precision, self.exponent)
        start_time = time.time()
        end_time = 0
        while end_time <= sample_time:
            self.resetInit(self.tree)
            sample, lp_sample = self.evaluate_error_at_sample(self.tree)
            values.append(sample)
            tmp_abs = abs(float(printMPFRExactly(lp_sample)) - sample)
            err.append(sample-float(printMPFRExactly(lp_sample)))
            abs_err.append(tmp_abs)
            rel_err.append(tmp_abs/sample)
            end_time = time.time() - start_time
        self.resetInit(self.tree)
        reset_default_precision()
        print("... Done with generation")
        return False, np.asarray(values), np.asarray(abs_err), np.asarray(rel_err), np.asarray(err)

    def elaborate_Gelpia_error_intervals(self, constraints, symbolic_affine):
        results=[]
        second_order_lower, second_order_upper = \
            SymbolicToGelpia(symbolic_affine.center, symbolic_affine.variables). \
                compute_concrete_bounds(debug=True, zero_output_epsilon=True)
        center_interval = Interval(second_order_lower, second_order_upper, True, True, digits_for_range)
        concrete_symbolic_interval = symbolic_affine.compute_interval_error(center_interval)
        results.append("Error domain 100%: [" + str(concrete_symbolic_interval.lower) + ", " +
                                                str(concrete_symbolic_interval.upper) + "]")
        for prob in constraints_probabilities:
            print("Error for prob: "+str(prob))
            constraint_dict = {}
            for constraint in constraints:
                values=constraints[constraint][prob]
                constraint_dict[str(constraint)]=[values[0], values[1]]
            constraints_interval = symbolic_affine.compute_interval_error(center_interval, constraints=constraint_dict)
            results.append("Error domain "+str(prob)+": [" + str(constraints_interval.lower) + ", " +
                       str(constraints_interval.upper) + "]")
        return results

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
            elif tree.root_name == "abs":
                return np.abs(sample_l), abs(mpfr(str(sample_l)))
            else:
                print("Operation not supported!")
                exit(-1)
        else:
            sample = tree.root_value[0].getSampleSet(n=1)[0]
            return sample, mpfr(str(sample))
