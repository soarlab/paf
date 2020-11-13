import copy

from pacal import ConstDistr
from pychebfun import *

from IntervalArithmeticLibrary import Interval, empty_interval_domain
import model
from SMT_Interface import create_exp_for_BinaryOperation_SMT_LIB
from linearprogramming import LP_Instance, LP_with_SMT
from mixedarithmetic import MixedArithmetic, PBox, from_PDFS_PBox_to_DSI, from_DSI_to_PBox, from_DSI_to_PBox_with_Delta, \
    from_CDFS_PBox_to_DSI
from plotting import plot_operation
from pruning import clean_co_domain
from regularizer import *
from project_utils import *
from gmpy2 import *

from setup_utils import global_interpolate, digits_for_cdf, discretization_points, divisions_SMT_pruning_error, \
    valid_for_exit_SMT_pruning_error, divisions_SMT_pruning_operation, valid_for_exit_SMT_pruning_operation, \
    recursion_limit_for_pruning_error, recursion_limit_for_pruning_operation


class quantizedPointMass:

    def __init__(self, wrapperInputDistribution, precision, exp):
        self.wrapperInputDistribution = wrapperInputDistribution
        self.precision = precision
        self.exp = exp
        set_context_precision(self.precision, self.exp)
        self.qValue = printMPFRExactly(mpfr(self.wrapperInputDistribution.discretization.affine.center.lower))
        reset_default_precision()
        self.name = self.qValue
        self.sampleInit = True
        self.distribution = ConstDistr(float(self.qValue))
        self.distribution.get_piecewise_pdf()
        self.discretization=None
        self.get_discretization()
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

    def get_discretization(self):
        if self.discretization==None:
            self.discretization=self.create_discretization()
            self.affine_error= self.createAffineErrorForValue()
        return self.discretization

    def createAffineErrorForValue(self):
        error=self.wrapperInputDistribution.discretization.affine.\
            perform_affine_operation("-", self.discretization.affine, dReal=False)
        return error

    def create_discretization(self):
        lower=self.wrapperInputDistribution.discretization.intervals[0].interval.lower
        upper=self.wrapperInputDistribution.discretization.intervals[-1].interval.upper
        with gmpy2.local_context(set_context_precision(self.precision, self.exp), round=gmpy2.RoundDown) as ctx:
            lower=round_number_down_to_digits(mpfr(lower),digits_for_discretization)
        with gmpy2.local_context(set_context_precision(self.precision, self.exp), round=gmpy2.RoundUp) as ctx:
            upper=round_number_up_to_digits(mpfr(upper),digits_for_discretization)
        return MixedArithmetic(lower,upper,
                               [PBox(Interval(lower,upper,True,True,digits_for_discretization),"0.0","1.0")])

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

    def __init__(self, leftoperand, operator, rightoperand, smt_triple, name, poly_precision, samples_dep_op,
                 exact_affine_forms=None, regularize=True,
                 convolution=True, dependent_mode="full_mc", is_error_computation=False):
        self.leftoperand = leftoperand
        self.operator = operator
        self.rightoperand = rightoperand
        self.name = name
        self.smt_triple=smt_triple
        self.poly_precision = poly_precision
        self.samples_dep_op = samples_dep_op
        self.regularize = regularize
        self.is_error_computation=is_error_computation
        self.convolution = convolution
        self.dependent_mode = dependent_mode
        self.exact_affines_forms=exact_affine_forms
        self.distribution = None
        self.distributionConv = None
        self.distributionSamp = None
        self.sampleInit = True
        self.discretization=None
        self.affine_error=None
        self.execute()

    def executeConvolution(self):
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
        #plt.figure()
        n, bins = np.histogram(tmp_res, bins='auto', density=True)
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

        expression_left=self.smt_triple[0]
        expression_right=self.smt_triple[1]
        smt_manager = self.smt_triple[2]

        expression_center= create_exp_for_BinaryOperation_SMT_LIB(expression_left,self.operator,expression_right)

        if self.is_error_computation:
            domain_affine_SMT = self.affine_error
            smt_manager.set_expression_left(expression_left, Interval(left_operand_discr_SMT.lower,left_operand_discr_SMT.upper,True,True, digits_for_discretization))
            smt_manager.set_expression_right(expression_right, Interval(right_operand_discr_SMT.lower,right_operand_discr_SMT.upper,True,True, digits_for_discretization))
            domain_affine_SMT.interval=\
                clean_co_domain(domain_affine_SMT.interval, smt_manager, expression_center,
                                divisions_SMT_pruning_error, valid_for_exit_SMT_pruning_error,
                                recursion_limit_for_pruning=recursion_limit_for_pruning_error,
                                start_recursion_limit=0, dReal=False)
            smt_manager.clean_expressions()
        else:
            domain_affine_SMT = left_operand_discr_SMT.affine.perform_affine_operation(self.operator,
                                                                                       right_operand_discr_SMT.affine)

        insides_SMT = []
        evaluation_points=set()
        print(self.leftoperand.name,self.operator,self.rightoperand.name)
        print("Pruning dependent operation...")

        for index_left, left_op_box_SMT in enumerate(left_operand_discr_SMT.intervals):
            for index_right, right_op_box_SMT in enumerate(right_operand_discr_SMT.intervals):

                smt_manager.set_expression_left(expression_left, left_op_box_SMT.interval)
                smt_manager.set_expression_right(expression_right, right_op_box_SMT.interval)
                domain_interval=left_op_box_SMT.interval.perform_interval_operation(self.operator,right_op_box_SMT.interval)
                intersection_interval = domain_interval.intersection(domain_affine_SMT.interval)
                if not intersection_interval == empty_interval_domain:
                    if smt_manager.check(debug=False, dReal=False):
                        #now we can clean the domain
                        clean_intersection_interval = \
                            clean_co_domain(intersection_interval,smt_manager,expression_center,
                                            (divisions_SMT_pruning_error if self.is_error_computation else divisions_SMT_pruning_operation),
                                            (valid_for_exit_SMT_pruning_error if self.is_error_computation else valid_for_exit_SMT_pruning_operation),
                                            recursion_limit_for_pruning=(recursion_limit_for_pruning_error if self.is_error_computation else recursion_limit_for_pruning_operation),
                                            start_recursion_limit=0, dReal=not self.is_error_computation)
                        inside_box_SMT = PBox(clean_intersection_interval,"prob","prob")
                        evaluation_points.add(Decimal(clean_intersection_interval.lower))
                        evaluation_points.add(Decimal(clean_intersection_interval.upper))
                        left_op_box_SMT.add_kid(inside_box_SMT)
                        right_op_box_SMT.add_kid(inside_box_SMT)
                        insides_SMT.append(inside_box_SMT)
                        smt_manager.clean_expressions()

        evaluation_points = sorted(evaluation_points)
        if len(evaluation_points)>discretization_points and not self.is_error_computation:
            step = round(len(evaluation_points) / discretization_points)
            step = max(1,step)
            evaluation_points = sorted(set(evaluation_points[::step]+[evaluation_points[-1]]))
        lp_inst_SMT=LP_with_SMT(self.leftoperand.name,self.rightoperand.name,
                    left_operand_discr_SMT.intervals,right_operand_discr_SMT.intervals,insides_SMT,evaluation_points)
        upper_bound_cdf_ind_SMT, upper_bound_cdf_val_SMT=lp_inst_SMT.optimize_max()
        lower_bound_cdf_ind_SMT, lower_bound_cdf_val_SMT=lp_inst_SMT.optimize_min()


        if not lower_bound_cdf_ind_SMT == upper_bound_cdf_ind_SMT:
            print("Lists should be identical")
            exit(-1)

        edge_cdf=lower_bound_cdf_ind_SMT
        val_cdf_low=lower_bound_cdf_val_SMT
        val_cdf_up=upper_bound_cdf_val_SMT
        pboxes=from_DSI_to_PBox(edge_cdf, val_cdf_low, edge_cdf, val_cdf_up)
        self.discretization =MixedArithmetic.clone_MixedArith_from_Args(domain_affine_SMT, pboxes)

    def executeDependent(self):
        if self.dependent_mode == "full_mc":
            self._full_mc_dependent_execution()
        elif self.dependent_mode == "analytic":
            self._analytic_dependent_execution()
        elif self.dependent_mode == "p-box":
            self._full_mc_dependent_execution()
            self._pbox_dependent_execution()

    def executeIndependent(self):
        self.executeConvolution()
        self.executeIndPBox()

    def executeIndPBox(self):
        left_op=copy.deepcopy(self.leftoperand.get_discretization())
        right_op=copy.deepcopy(self.rightoperand.get_discretization())
        domain_affine = left_op.affine.perform_affine_operation(self.operator, right_op.affine)
        insiders=[]
        evaluation_points=set()
        print(self.leftoperand.name,self.operator,self.rightoperand.name)

        print("Left:\n", self.probability_in_insiders(self.leftoperand.discretization.intervals))
        print("Right:\n", self.probability_in_insiders(self.rightoperand.discretization.intervals))

        for index_left, left_op_box in enumerate(left_op.intervals):
            pdf_left = Decimal(left_op_box.cdf_up)-Decimal(left_op_box.cdf_low)
            for index_right, right_op_box in enumerate(right_op.intervals):
                pdf_right = Decimal(right_op_box.cdf_up) - Decimal(right_op_box.cdf_low)
                domain_interval=left_op_box.interval.perform_interval_operation(self.operator,right_op_box.interval)
                probability=pdf_left*pdf_right
                #probability=round_near(probability,digits_for_cdf)
                inside_box = PBox(domain_interval, dec2Str(probability), dec2Str(probability))
                insiders.append(inside_box)
                evaluation_points.add(Decimal(domain_interval.lower))
                evaluation_points.add(Decimal(domain_interval.upper))

        print("Result:\n", self.probability_in_insiders(insiders))

        evaluation_points = sorted(evaluation_points)
        if len(evaluation_points)>discretization_points and not self.is_error_computation:
            step = round(len(evaluation_points) / discretization_points)
            step = max(1,step)
            evaluation_points = sorted(set(evaluation_points[::step]+[evaluation_points[-1]]))

        edge_cdf, val_cdf_low, val_cdf_up=from_PDFS_PBox_to_DSI(insiders, evaluation_points)
        pboxes = from_DSI_to_PBox(edge_cdf, val_cdf_low, edge_cdf, val_cdf_up)

        self.discretization = MixedArithmetic.clone_MixedArith_from_Args(domain_affine, pboxes)

    def probability_in_insiders(self, insiders):
        res_left= Decimal("0")
        res_right = Decimal("0")
        for inside in insiders:
            res_left=res_left+Decimal(inside.cdf_low)
            res_right=res_right+Decimal(inside.cdf_up)
        ret="Total probability in insiders low: "+dec2Str(res_left)+"\n"
        ret = ret+"Total probability in insiders up: " + dec2Str(res_right)
        return ret

    def execute(self):
        if self.distribution == None:
            if self.is_error_computation:
                self.affine_error=self.rightoperand.affine_error
            else:
                self.compute_error_affine_form()
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

    def compute_error_affine_form(self):
        if self.operator == "+":
            self.affine_error = self.leftoperand.affine_error.perform_affine_operation\
                ("+",self.rightoperand.affine_error)
        elif self.operator == "-":
            self.affine_error = self.leftoperand.affine_error.perform_affine_operation \
                ("-", self.rightoperand.affine_error)
        elif self.operator == "*":
            x_erry = self.exact_affines_forms[0].\
                perform_affine_operation("*", self.rightoperand.affine_error,
                                         recursion_limit_for_pruning=recursion_limit_for_pruning_error, dReal=False)
            y_errx=self.exact_affines_forms[1].\
                perform_affine_operation("*", self.leftoperand.affine_error,
                                         recursion_limit_for_pruning=recursion_limit_for_pruning_error, dReal=False)
            errx_erry=self.leftoperand.affine_error.\
                perform_affine_operation("*", self.rightoperand.affine_error,
                                         recursion_limit_for_pruning=recursion_limit_for_pruning_error, dReal=False)
            self.affine_error=x_erry.perform_affine_operation("+",
                              y_errx.perform_affine_operation("+", errx_erry))
        elif self.operator == "/":
            x_erry = self.exact_affines_forms[0].\
                perform_affine_operation("/", self.rightoperand.affine_error,
                                         recursion_limit_for_pruning=recursion_limit_for_pruning_error, dReal=False)
            y_errx = self.exact_affines_forms[1].\
                perform_affine_operation("/", self.leftoperand.affine_error,
                                         recursion_limit_for_pruning=recursion_limit_for_pruning_error, dReal=False)
            errx_erry = self.leftoperand.affine_error.\
                perform_affine_operation("/", self.rightoperand.affine_error,
                                         recursion_limit_for_pruning=recursion_limit_for_pruning_error, dReal=False)
            self.affine_error = x_erry.perform_arithmetic_operation("+",
                                y_errx.perform_arithmetic_operation("+", errx_erry))

        elif self.operator == "*+":
            self.affine_error = self.leftoperand.affine_error.\
                perform_affine_operation("+",
                        self.exact_affines_forms[0].perform_affine_operation("+", self.leftoperand.affine_error).
                        perform_affine_operation("*",self.rightoperand.discretization.affine))
        else:
            print("Operation not supported!")
            exit(-1)

    def get_discretization(self):
        if self.discretization==None:
            self.execute()
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
            self.distribution = model.abs(operand)
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
        self.discretization=None
        self.affine_error=None
        self.get_discretization()

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
        if self.discretization==None and self.affine_error == None:
            self.discretization = self.distribution.get_discretization()
            self.affine_error = self.distribution.affine_error
        return self.discretization

    #if self.discretization == None and self.affine_error == None:
    #    self.discretization = createDSIfromDistribution(self, n=discretization_points)