import copy
import math
from pychebfun import *

import IntervalArithmetic
import model
import pbox
from linearprogramming import LP_Instance
from pbox import createDSIfromDistribution, PBox
from regularizer import *
from project_utils import *
from gmpy2 import *

from setup_utils import global_interpolate, discretization_points


class quantizedPointMass:

    def __init__(self, wrapperInputDistribution, precision, exp):
        self.wrapperInputDistribution = wrapperInputDistribution
        self.inputdistribution = self.wrapperInputDistribution.execute()
        self.precision = precision
        self.exp = exp
        set_context_precision(self.precision, self.exp)
        qValue = printMPFRExactly(mpfr(str(self.inputdistribution.rand(1)[0])))
        reset_default_precision()
        self.name = self.inputdistribution.getName()
        self.sampleInit = True
        self.distribution = ConstDistr(float(qValue))
        self.distribution.get_piecewise_pdf()
        self.a = self.distribution.range_()[0]
        self.b = self.distribution.range_()[-1]

    def execute(self):
        return self.distribution

    def resetSampleInit(self):
        self.sampleInit = True

    def getSampleSet(self, n=100000):
        # it remembers values for future operations
        if self.sampleInit:
            self.sampleSet = self.distribution.rand(n)
            self.sampleInit = False
        return self.sampleSet

    def getName(self):
        return self.name


class DependentOperationExecutor(object):
    def __init__(self, bins, n, interp_points):
        self.bins = bins
        self.n = n
        self.interp_points = interp_points
        self.name = "Dep. Operation: bins = " + str(self.bins) + ", values = " + str(self.n) + "]"
        self.interp_dep_op = chebfun(self.executeOperation, domain=[min(bins), max(bins)], N=self.interp_points)

    def executeOperation(self, t):
        if isinstance(t, float) or isinstance(t, int) or len(t) == 1:
            if t < min(self.bins) or t > max(self.bins):
                return 0.0
            else:
                index_bin = np.digitize(t, self.bins)
                return abs(self.n[index_bin])
        else:
            res = np.zeros(len(t))
            tis = t
            for index, ti in enumerate(tis):
                if ti < min(self.bins) or ti > max(self.bins):
                    res[index] = 0.0
                else:
                    index_bin = np.digitize(ti, self.bins, right=True)
                    res[index] = self.n[index_bin - 1]
            return abs(res)

    def __getstate__(self):
        tmp_dict = self.__dict__  # get attribute dictionary
        if 'interp_dep_op' in tmp_dict:
            del tmp_dict['interp_dep_op']  # remove interp_trunc_norm entry
        return tmp_dict
        # restore object state from data representation generated
        # by __getstate__

    def __setstate__(self, dict):
        self.bins = dict["bins"]
        self.n = dict["n"]
        self.interp_points = dict["interp_points"]
        self.name = dict["name"]
        if 'interp_dep_op' not in dict:
            dict['interp_dep_op'] = chebfun(self.executeOperation, domain=[min(self.bins), max(self.bins)],
                                            N=self.interp_points)
        self.__dict__ = dict  # make dict our attribute dictionary

    def __call__(self, t, *args, **kwargs):
        return self.interp_dep_op(t)


class BinOpDist:
    """
    Wrapper class for the result of an arithmetic operation on PaCal distributions
    """

    def __init__(self, leftoperand, operator, rightoperand, smt_triple, name, poly_precision, samples_dep_op, regularize=True,
                 convolution=True, dependent_mode="full_mc"):
        self.leftoperand = leftoperand
        self.operator = operator
        self.rightoperand = rightoperand
        self.name = name
        self.smt_triple=smt_triple
        self.poly_precision = poly_precision
        self.samples_dep_op = samples_dep_op
        self.regularize = regularize
        self.convolution = convolution
        self.dependent_mode = dependent_mode
        self.distribution = None
        self.distributionConv = None
        self.distributionSamp = None
        self.sampleInit = True
        self.discretization=[]
        self.execute()

    def executeIndependent(self):
        if self.operator == "+":
            self.distributionConv = self.leftoperand.execute() + self.rightoperand.execute()
        elif self.operator == "-":
            self.distributionConv = self.leftoperand.execute() - self.rightoperand.execute()
        elif self.operator == "*":
            self.distributionConv = self.leftoperand.execute() * self.rightoperand.execute()
        elif self.operator == "/":
            self.distributionConv = self.leftoperand.execute() / self.rightoperand.execute()
        # operator to multiply by a relative error
        elif self.operator == "*+":
            self.distributionConv = self.leftoperand.execute() * (
                    1.0 + (self.rightoperand.unit_roundoff * self.rightoperand.execute()))
        else:
            print("Operation not supported!")
            exit(-1)

        self.distributionConv.get_piecewise_pdf()

        if self.regularize:
            self.distributionConv = chebfunInterpDistr(self.distributionConv, 10)
            self.distributionConv = normalizeDistribution(self.distributionConv)

        self.aConv = self.distributionConv.range_()[0]
        self.bConv = self.distributionConv.range_()[-1]

    def operationDependent(self, elaborateBorders):
        leftOp = self.leftoperand.getSampleSet(self.samples_dep_op)
        rightOp = self.rightoperand.getSampleSet(self.samples_dep_op)

        if self.operator == "*+":
            res = np.array(leftOp) * (1 + (self.rightoperand.unit_roundoff * np.array(rightOp)))
            if elaborateBorders:
                res = self.elaborateBorders(leftOp, self.operator,
                                            (1 + (self.rightoperand.unit_roundoff * np.array(rightOp))),
                                            res)
        else:
            res = eval("np.array(leftOp)" + self.operator + "np.array(rightOp)")
            if elaborateBorders:
                res = self.elaborateBorders(leftOp, self.operator, rightOp, res)

        return res

    def elaborateBorders(self, leftOp, operator, rightOp, res):
        x1 = min(leftOp)
        x2 = max(leftOp)
        y1 = min(rightOp)
        y2 = max(rightOp)
        tmp_res = []
        for tmp_1 in [x1, x2]:
            for tmp_2 in [y1, y2]:
                tmp_res.append(eval(str(tmp_1) + operator + str(tmp_2)))
        res[-1] = min(tmp_res)
        res[-2] = max(tmp_res)
        return res

    def _full_mc_dependent_execution(self):
        tmp_res = self.distributionValues
        bin_nb = int(math.ceil(math.sqrt(len(tmp_res))))

        # !!!!!!!!!!!!!!!!!!!!!!!!!!
        # Try also with bins=AUTO !!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!

        n, bins, patches = plt.hist(tmp_res, bins='auto', density=True)

        breaks = [min(bins), max(bins)]

        self.distributionSamp = MyFunDistr(self.name, DependentOperationExecutor(bins, n, self.poly_precision), breakPoints=breaks,
                                           interpolated=global_interpolate)
        self.distributionSamp.get_piecewise_pdf()

        if self.regularize:
            self.distributionSamp = chebfunInterpDistr(self.distributionSamp, 10)
            self.distributionSamp = normalizeDistribution(self.distributionSamp, init=True)

        self.aSamp = self.distributionSamp.range_()[0]
        self.bSamp = self.distributionSamp.range_()[-1]

    def _pbox_dependent_execution(self):
        left_operand_discr_SMT=copy.deepcopy(self.leftoperand.get_discretization())
        right_operand_discr_SMT=copy.deepcopy(self.rightoperand.get_discretization())

        left_operand_discr_INT=copy.deepcopy(self.leftoperand.get_discretization())
        right_operand_discr_INT=copy.deepcopy(self.rightoperand.get_discretization())

        expression_left=self.smt_triple[0]
        expression_right=self.smt_triple[1]
        smt_manager = self.smt_triple[2]

        insides_SMT = []
        insides_INT = []

        for index_left, left_op_box_SMT in enumerate(left_operand_discr_SMT):
            for index_right, right_op_box_SMT in enumerate(right_operand_discr_SMT):

                smt_manager.set_expression_left(expression_left, left_op_box_SMT.lower, left_op_box_SMT.upper )
                smt_manager.set_expression_right(expression_right, right_op_box_SMT.lower, right_op_box_SMT.upper )
                low, sup=IntervalArithmetic.perform_interval_operation(left_op_box_SMT.lower, left_op_box_SMT.upper, self.operator,
                                                              right_op_box_SMT.lower, right_op_box_SMT.upper)

                if smt_manager.check(debug=True):
                    inside_box_SMT= PBox(low,sup,"prob")
                    inside_box_SMT.is_marginal=False
                    left_op_box_SMT.add_kid(inside_box_SMT)
                    right_op_box_SMT.add_kid(inside_box_SMT)
                    insides_SMT.append(inside_box_SMT)

                inside_box_INT = PBox(low, sup, "prob")
                inside_box_INT.is_marginal = False
                left_operand_discr_INT[index_left].add_kid(inside_box_INT)
                right_operand_discr_INT[index_right].add_kid(inside_box_INT)
                insides_INT.append(inside_box_INT)

        lp_inst_SMT=LP_Instance(left_operand_discr_SMT, right_operand_discr_SMT, insides_SMT)
        upper_bound_cdf_ind_SMT, upper_bound_cdf_val_SMT=lp_inst_SMT.optimize_max()
        lower_bound_cdf_ind_SMT, lower_bound_cdf_val_SMT=lp_inst_SMT.optimize_min()

        lp_inst_INT=LP_Instance(left_operand_discr_INT, right_operand_discr_INT, insides_INT)
        upper_bound_cdf_ind_INT, upper_bound_cdf_val_INT=lp_inst_INT.optimize_max()
        lower_bound_cdf_ind_INT, lower_bound_cdf_val_INT=lp_inst_INT.optimize_min()

        plt.figure()
        plt.plot(lower_bound_cdf_ind_SMT, lower_bound_cdf_val_SMT, '-o', c="red", label="lower_bound_SMT")
        plt.plot(upper_bound_cdf_ind_SMT, upper_bound_cdf_val_SMT, '-o', c="purple", label="upper_bound_SMT")

        plt.plot(upper_bound_cdf_ind_INT, upper_bound_cdf_val_INT, '-o', c="blue", label="lower_bound_INT")
        plt.plot(lower_bound_cdf_ind_INT, lower_bound_cdf_val_INT, '-o', c="black", label="upper_bound_INT")

        plt.legend()
        plt.show()

        lower_bound_dst=pbox.createDiscreteDistrLower("lb",lower_bound_cdf_ind_SMT, lower_bound_cdf_val_SMT)
        upper_bound_dst=pbox.createDiscreteDistrUpper("ub",upper_bound_cdf_ind_SMT, upper_bound_cdf_val_SMT)

        return

    def _analytic_dependent_execution(self):
        """ Compute the dependent operation by integrating over all variables"""
        # find set of variables
        self.variable_dictionary = {}
        self._populate_variable_dictionary(self.leftoperand)
        self._populate_variable_dictionary(self.rightoperand)
        # this is only guaranteed to work from Python 3.7
        variable_tuple = list(self.variable_dictionary)
        variable_nb = len(variable_tuple)
        # find integration bounds
        lower_bound = np.full(variable_nb, np.NINF)
        upper_bound = np.full(variable_nb, np.inf)
        for i in range(variable_nb):
            lower_bound[i] = self.variable_dictionary[variable_tuple[i]].range_()[0]
            upper_bound[i] = self.variable_dictionary[variable_tuple[i]].range_()[1]
        # perform multidimensional integration

    def _nd_trap(self, f, range_array, value_array, auxiliary_array):
        """ Recursively performs an n-dimensional integration using the trapezoidal rule
            If f is integrated along n dimensions, then at any call to _nd_trap the array range_array will have
            dimension 1 <= m < n and value_array will have dimension n-m.
            auxiliary_array is the list of inputs of f which are not integrated against (can be empty!).
        """
        integral = 0
        nb_steps = 10.0
        if len(range_array) == 0:
            return integral
        h = (range_array[0][1] - range_array[0][0]) / nb_steps
        if len(range_array) == 1:
            x = value_array
            x.append(range_array[0][0])
            integral += 0.5 * f(x, auxiliary_array)
            for i in range(1, nb_steps):
                x = x[:-1]
                x.append(range_array[0][0] + (i * h))
                integral += f(x, auxiliary_array)
            x = x[:-1]
            x.append(range_array[0][1])
            integral += 0.5 * f(x, auxiliary_array)
            return h * integral
        else:
            x = value_array
            x.append(range_array[0][0])
            integral += 0.5 * self._nd_trap(self, f, range_array[1:], x, auxiliary_array)
            for i in range(0, nb_steps):
                x = x[:-1]
                x.append(range_array[0][0] + (i * h))
                integral += self._nd_trap(self, f, range_array[1:], x, auxiliary_array)
            x = x[:-1]
            x.append(range_array[0][1])
            integral += 0.5 * self._nd_trap(self, f, range_array[1:], x, auxiliary_array)
            return h * integral

    def _evaluate_tree(self, input_tuple, t):
        """ Function to be integrated in the analytic method """

    def _populate_variable_dictionary(self, tree, variable_dictionary):
        if tree is not None:
            # test if tree is a leaf
            if tree.left is None:
                # test if it contains a variable
                if not tree.root_value.isScalar:
                    # test if it is already contained in the set
                    if tree.root_name not in variable_dictionary:
                        variable_dictionary[tree.root_name] = tree.root_value.distribution
            # else recursively call _populate_variable_set
            else:
                self._populate_variable_dictionary(tree.left, variable_dictionary)
                self._populate_variable_dictionary(tree.right, variable_dictionary)

    def executeDependent(self):
        if self.dependent_mode == "full_mc":
            self._full_mc_dependent_execution()
        elif self.dependent_mode == "analytic":
            self._analytic_dependent_execution()
        elif self.dependent_mode == "p-box":
            self._full_mc_dependent_execution()
            self._pbox_dependent_execution()

    def execute(self):
        if self.distribution == None:
            if self.convolution:
                self.executeIndependent()
                self.distributionValues = self.operationDependent(elaborateBorders=False)
                self.distribution = self.distributionConv
                self.a = self.aConv
                self.b = self.bConv
            else:
                self.distributionValues = self.operationDependent(elaborateBorders=False)
                self.executeDependent()
                self.distribution = self.distributionSamp
                self.a = self.aSamp
                self.b = self.bSamp

            self.distribution.get_piecewise_pdf()
        return self.distribution

    def get_discretization(self):
        if len(self.discretization)==0:
            self.discretization = createDSIfromDistribution(self.distribution, n=discretization_points)
        return self.discretization

    def getSampleSet(self, n=100000):
        # it remembers values for future operations
        if self.sampleInit:
            self.execute()
            self.sampleSet = self.distributionValues
            self.sampleInit = False
        return self.sampleSet

    def resetSampleInit(self):
        self.sampleInit = True

    def getName(self):
        return self.name


class UnOpDist:
    """
    Wrapper class for the result of unary operation on a PaCal distribution
    """

    def __init__(self, operand, name, operation=None):
        if operation is None:
            self.distribution = operand.execute()
        elif operation is "exp":
            self.distribution = model.exp(operand.execute())
            self.distribution.get_piecewise_pdf()
        elif operation is "cos":
            self.distribution = model.cos(operand.execute())
            self.distribution.get_piecewise_pdf()
        elif operation is "sin":
            self.distribution = model.sin(operand.execute())
            self.distribution.get_piecewise_pdf()
        elif operation is "abs":
            self.distribution = model.abs(operand.execute())
            self.distribution.get_piecewise_pdf()
        else:
            print("Unary operation not yet supported")
            exit(-1)

        self.operand = operand
        self.name = name
        self.operation=operation
        self.sampleInit=True
        self.a = self.distribution.range_()[0]
        self.b = self.distribution.range_()[-1]

    def execute(self):
        return self.distribution

    def resetSampleInit(self):
        self.sampleInit = True
        self.operand.resetSampleInit()

    def getSampleSet(self, n=100000):
        # it remembers values for future operations
        if self.sampleInit:
            self.sampleSet = self.operand.getSampleSet(n)
            self.sampleInit = False
            if self.operation is "exp":
                self.sampleSet = np.exp(self.sampleSet)
            elif self.operation is "cos":
                self.sampleSet = np.cos(self.sampleSet)
            elif self.operation is "sin":
                self.sampleSet = np.sin(self.sampleSet)
            elif self.operation is "abs":
                self.sampleSet = np.abs(self.sampleSet)
        return self.sampleSet

    def getName(self):
        return self.name

    def get_discretization(self):
        return self.distribution.get_discretization()