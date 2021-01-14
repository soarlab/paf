import copy
import multiprocessing as mp
from multiprocessing.pool import Pool

from pacal import ConstDistr
from pychebfun import *
from sympy.plotting.intervalmath import interval

from AffineArithmeticLibrary import AffineInstance
from IntervalArithmeticLibrary import Interval, empty_interval, check_zero_is_in_interval, find_min_abs_interval, \
    check_sterbenz_apply

#from model import BoundingPair
import model
from SMT_Interface import create_exp_for_BinaryOperation_SMT_LIB
from SymbolicAffineArithmetic import SymbolicAffineInstance, SymbolicAffineManager, SymExpression, SymbolicToGelpia
from linearprogramming import LP_with_SMT
from mixedarithmetic import MixedArithmetic, PBox, from_PDFS_PBox_to_DSI, from_DSI_to_PBox
from plotting import plot_operation
from pruning import clean_co_domain
from regularizer import *
from project_utils import *
from gmpy2 import *

from setup_utils import global_interpolate, digits_for_input_cdf, discretization_points, divisions_SMT_pruning_error, \
    valid_for_exit_SMT_pruning_error, divisions_SMT_pruning_operation, valid_for_exit_SMT_pruning_operation, \
    recursion_limit_for_pruning_error, recursion_limit_for_pruning_operation, num_processes, \
    num_processes_dependent_operation, round_constants_to_nearest, MyPool


import matplotlib.pyplot as plt
import matplotlib


def dependentIteration(index_left, index_right, smt_manager_input, expression_left, expression_center, expression_right,
                       operator, left_op_box_SMT, right_op_box_SMT, domain_affine_SMT, error_computation,
                       symbolic_affine_form, concrete_symbolic_interval, constraint_expression, center_interval):
    #print("Start Square_"+str(index_left)+"_"+str(index_right))
    smt_manager = copy.deepcopy(smt_manager_input)
    smt_manager.set_expression_left(expression_left, left_op_box_SMT.interval)
    smt_manager.set_expression_right(expression_right, right_op_box_SMT.interval)
    domain_interval = left_op_box_SMT.interval.perform_interval_operation(operator, right_op_box_SMT.interval)
    intersection_interval = domain_interval.intersection(domain_affine_SMT.interval)
    if not intersection_interval == empty_interval:
        z3=smt_manager.check(debug=False, dReal=False)
        dreal=smt_manager.check(debug=False, dReal=True)
        solver_res= (z3 and dreal)
        if solver_res:
            # now we can clean the domain
            if error_computation:
                intersection_interval = intersection_interval.intersection(concrete_symbolic_interval)
                gelpia_interval=right_op_box_SMT.interval.intersection(left_op_box_SMT.interval)
                if gelpia_interval==empty_interval:
                    #In case the two intervals do not overlap it means they are not related
                    return [None, None, empty_interval]
                constraint_dict = {
                    str(constraint_expression): [gelpia_interval.lower, gelpia_interval.upper]}
                constraints_interval = symbolic_affine_form.compute_interval_error(center_interval,constraints=constraint_dict)
                print(index_left, index_right, "Interval Left: " + left_op_box_SMT.interval.lower + " " + left_op_box_SMT.interval.upper)
                print(index_left, index_right, "Interval Right: " + right_op_box_SMT.interval.lower + " " + right_op_box_SMT.interval.upper)
                print(index_left, index_right, "Gelpia Interval: " + gelpia_interval.lower + " " + gelpia_interval.upper)
                print(index_left, index_right, "Error from Gelpia : ["+str(constraints_interval.lower)+","+str(constraints_interval.upper)+"]")
                print(index_left, index_right, "Intersection Interval: "+ intersection_interval.lower + " "+ intersection_interval.upper)
                intersection_interval = intersection_interval.intersection(constraints_interval)
                print(index_left, index_right, "Final Error: ["+str(intersection_interval.lower)+","+str(intersection_interval.upper)+"]")
                return [index_left, index_right, intersection_interval]

            if z3>1 and dreal>1:
                return [index_left, index_right, intersection_interval]

            clean_intersection_interval = \
                clean_co_domain(intersection_interval, smt_manager, expression_center,
                                (divisions_SMT_pruning_error if error_computation else divisions_SMT_pruning_operation),
                                (valid_for_exit_SMT_pruning_error if error_computation else valid_for_exit_SMT_pruning_operation),
                                recursion_limit_for_pruning=(recursion_limit_for_pruning_error if error_computation else recursion_limit_for_pruning_operation),
                                start_recursion_limit=0, dReal=not error_computation)
            #print("Done Pruning Square_" + str(index_left) + "_" + str(index_right))
            return [index_left,index_right,clean_intersection_interval]
    #print("Done Affine Square_" + str(index_left) + "_" + str(index_right))
    return [None, None, empty_interval]

class ConstantManager:
    i=1

    def __init__(self):
        print("Constant Manager should never be instantiated")
        raise NotImplementedError

    @staticmethod
    def get_new_constant_index():
        tmp=ConstantManager.i
        ConstantManager.i= ConstantManager.i + 1
        return "Constant_"+str(tmp)

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
        self.discretization = None
        self.affine_error = None
        self.symbolic_error = None
        self.symbolic_affine=None
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
        if self.discretization is None:
            self.discretization = self.create_discretization()
            self.affine_error = self.createAffineErrorForValue()
            self.symbolic_affine = self.createSymbolicAffineInstance()
            self.symbolic_error = self.wrapperInputDistribution.symbolic_affine.\
                perform_affine_operation("-", self.symbolic_affine)
        return self.discretization

    def createSymbolicAffineInstance(self):
        sym_term=SymExpression("["+self.discretization.lower+","+self.discretization.upper+"]")
        return SymbolicAffineInstance(sym_term, {}, {})

    def createAffineErrorForValue(self):
        error=self.wrapperInputDistribution.discretization.affine.\
            perform_affine_operation("-", self.discretization.affine, dReal=False)
        return error

    def create_discretization(self):
        lower = self.wrapperInputDistribution.discretization.intervals[0].interval.lower
        upper = self.wrapperInputDistribution.discretization.intervals[-1].interval.upper
        with gmpy2.local_context(set_context_precision(self.precision, self.exp),
                                 round=(gmpy2.RoundDown if not round_constants_to_nearest else gmpy2.RoundToNearest)) as ctx:
            lower=round_number_down_to_digits(mpfr(lower), digits_for_range)

        with gmpy2.local_context(set_context_precision(self.precision, self.exp),
                                 round=(gmpy2.RoundUp if not round_constants_to_nearest else gmpy2.RoundToNearest)) as ctx:
            upper=round_number_up_to_digits(mpfr(upper), digits_for_range)
        #The following somehow remind of a dirac distribution.
        return MixedArithmetic(lower, upper,
                               [PBox(Interval(lower, upper, True, True, digits_for_range), "0.0", "1.0")])

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
        self.discretization = None
        self.affine_error = None
        self.do_quantize_operation = True
        self.symbolic_affine = None
        self.symbolic_error = None
        self.bounding_pair = None
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

    def operationDependent(self):
        leftOp = self.leftoperand.getSampleSet(self.samples_dep_op)
        rightOp = self.rightoperand.getSampleSet(self.samples_dep_op)

        if self.operator == "*+":
            res = np.array(leftOp) * (1 + (self.rightoperand.unit_roundoff * np.array(rightOp)))
        else:
            res = eval("np.array(leftOp)" + self.operator + "np.array(rightOp)")

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
        left_operand_discr_SMT = copy.deepcopy(self.leftoperand.get_discretization())
        right_operand_discr_SMT = copy.deepcopy(self.rightoperand.get_discretization())

        expression_left=self.smt_triple[0]
        expression_right=self.smt_triple[1]
        smt_manager = self.smt_triple[2]

        expression_center= create_exp_for_BinaryOperation_SMT_LIB(expression_left,self.operator,expression_right)

        if self.is_error_computation:
            self.affine_error.update_interval()
            domain_affine_SMT = self.affine_error
            smt_manager.clean_expressions()
            self.symbolic_affine = self.symbolic_error
            constraint_expression=self.rightoperand.name #symbolic_affine.center
            second_order_lower, second_order_upper = \
                SymbolicToGelpia(self.symbolic_affine.center,self.symbolic_affine.variables).\
                    compute_concrete_bounds(zero_output_epsilon=True)
            center_interval = Interval(second_order_lower, second_order_upper, True, True, digits_for_range)
            concrete_symbolic_interval = self.symbolic_affine.compute_interval_error(center_interval)
            print("Error domain: ["+str(concrete_symbolic_interval.lower)+", "+str(concrete_symbolic_interval.upper)+"]")
        else:
            domain_affine_SMT = left_operand_discr_SMT.affine.perform_affine_operation(self.operator,
                                                                                       right_operand_discr_SMT.affine)
            self.symbolic_affine = None #self.leftoperand.symbolic_affine.perform_affine_operation(self.operator,
                                                                                   # self.rightoperand.symbolic_affine)
            constraint_expression=None
            center_interval=None
            concrete_symbolic_interval = None #self.symbolic_affine.compute_interval()
        insides_SMT = []
        tmp_insides_SMT = []

        evaluation_points=set()
        print(self.leftoperand.name,self.operator,self.rightoperand.name)
        print("Left-Intervals: "+str(len(left_operand_discr_SMT.intervals)))
        print("Right-Intervals: "+str(len(right_operand_discr_SMT.intervals)))
        print("Pruning dependent operation...")

        pool = MyPool(processes=num_processes_dependent_operation)#, maxtasksperchild=3)
        tmp_results=[]

        for index_left, left_op_box_SMT in enumerate(left_operand_discr_SMT.intervals):
            for index_right, right_op_box_SMT in enumerate(right_operand_discr_SMT.intervals):
                domain_interval = left_op_box_SMT.\
                    interval.perform_interval_operation(self.operator,right_op_box_SMT.interval)
                intersection_interval = domain_interval.intersection(domain_affine_SMT.interval)
                if not intersection_interval == empty_interval:
                    #intersection_interval = intersection_interval.intersection(concrete_symbolic_interval)
                    if not intersection_interval == empty_interval:
                        tmp_results.append(
                            pool.apply_async(dependentIteration,
                                args=[index_left, index_right, smt_manager, expression_left,
                                    expression_center, expression_right, self.operator, left_op_box_SMT,
                                    right_op_box_SMT, domain_affine_SMT, self.is_error_computation,
                                    self.symbolic_affine, concrete_symbolic_interval,
                                    constraint_expression, center_interval],
                                callback=tmp_insides_SMT.append))
        print("Number of jobs for dependent operation: "+str(len(tmp_results)))
        pool.close()
        pool.join()
        print("\nDone with dependent operation\n")

        for triple in tmp_insides_SMT:
            if not triple[2] == empty_interval:
                inside_box_SMT = PBox(triple[2], "prob", "prob")
                insides_SMT.append(inside_box_SMT)
                left_operand_discr_SMT.intervals[triple[0]].add_kid(inside_box_SMT)
                right_operand_discr_SMT.intervals[triple[1]].add_kid(inside_box_SMT)
                evaluation_points.add(Decimal(inside_box_SMT.interval.lower))
                evaluation_points.add(Decimal(inside_box_SMT.interval.upper))

        evaluation_points = sorted(evaluation_points)

        if len(evaluation_points) > discretization_points and not self.is_error_computation:
            step = round(len(evaluation_points) / discretization_points)
            step = max(1, step)
            evaluation_points = sorted(set(evaluation_points[::step]+[evaluation_points[-1]]))

        lp_inst_SMT=LP_with_SMT(self.leftoperand.name,self.rightoperand.name,
                    left_operand_discr_SMT.intervals,right_operand_discr_SMT.intervals,insides_SMT,evaluation_points)
        upper_bound_cdf_ind_SMT, upper_bound_cdf_val_SMT=lp_inst_SMT.optimize_max()
        lower_bound_cdf_ind_SMT, lower_bound_cdf_val_SMT=lp_inst_SMT.optimize_min()

        print("Done with LP optimization problem")

        if not lower_bound_cdf_ind_SMT == upper_bound_cdf_ind_SMT:
            print("Lists should be identical")
            exit(-1)

        edge_cdf=lower_bound_cdf_ind_SMT
        val_cdf_low=lower_bound_cdf_val_SMT
        val_cdf_up=upper_bound_cdf_val_SMT
        pboxes=from_DSI_to_PBox(edge_cdf, val_cdf_low, edge_cdf, val_cdf_up)
        self.discretization = MixedArithmetic.clone_MixedArith_from_Args(domain_affine_SMT, pboxes)


    def executeDependent(self):
        if self.dependent_mode == "full_mc":
            self._full_mc_dependent_execution()
        elif self.dependent_mode == "p-box":
            self._full_mc_dependent_execution()
            self._pbox_dependent_execution()

    def executeIndependent(self):
        self.executeConvolution()
        self.executeIndPBox()
        bounding_pair_operation = BoundingPairOperation(self.operator, self.leftoperand, self.rightoperand)
        bounding_pair_operation.perform_operation()
        self.bounding_pair = bounding_pair_operation.output

    def executeIndPBox(self):
        left_op=copy.deepcopy(self.leftoperand.get_discretization())
        right_op=copy.deepcopy(self.rightoperand.get_discretization())
        domain_affine = left_op.affine.perform_affine_operation(self.operator, right_op.affine)
        self.symbolic_affine = None #self.leftoperand.symbolic_affine.perform_affine_operation\
                                        #(self.operator, self.rightoperand.symbolic_affine)
        insiders=[]
        evaluation_points=set()

        print(self.leftoperand.name,self.operator,self.rightoperand.name)

        print("Left:\n", self.probability_in_insiders(self.leftoperand.discretization.intervals))
        print("Right:\n", self.probability_in_insiders(self.rightoperand.discretization.intervals))

        for index_left, left_op_box in enumerate(left_op.intervals):
            pdf_left = Decimal(left_op_box.cdf_up) - Decimal(left_op_box.cdf_low)
            #left_upper=Interval(left_op_box.cdf_up, left_op_box.cdf_up, True, True, digits_for_cdf)
            #left_lower=Interval(left_op_box.cdf_low, left_op_box.cdf_low, True, True, digits_for_cdf)
            #pdf_left=left_upper.perform_interval_operation("-", left_lower)
            for index_right, right_op_box in enumerate(right_op.intervals):
                pdf_right = Decimal(right_op_box.cdf_up) - Decimal(right_op_box.cdf_low)
                #right_upper = Interval(right_op_box.cdf_up, right_op_box.cdf_up, True, True, digits_for_cdf)
                #right_lower = Interval(right_op_box.cdf_low, right_op_box.cdf_low, True, True, digits_for_cdf)
                #pdf_right = right_upper.perform_interval_operation("-", right_lower)
                #probability_interval=pdf_left.perform_interval_operation("*", pdf_right)
                probability_value = pdf_left * pdf_right
                domain_interval=left_op_box.interval.perform_interval_operation(self.operator,right_op_box.interval)
                inside_box = PBox(domain_interval, dec2Str(probability_value), dec2Str(probability_value))
                #inside_box = PBox(domain_interval, probability_interval.lower, probability_interval.upper)
                insiders.append(inside_box)
                evaluation_points.add(Decimal(domain_interval.lower))
                evaluation_points.add(Decimal(domain_interval.upper))

        print("Potential error of the operation:\n", self.probability_in_insiders(insiders))
        res_left = Decimal("0")
        for inside in insiders:
            res_left = res_left + Decimal(inside.cdf_low)
        ret = "Check Total probability in insiders: " + dec2Str(res_left) + "\n"
        print(ret)

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
        ret="Check Total probability in insiders: "+dec2Str(res_right-res_left)+"\n"
        return ret

    def execute(self):
        if self.distribution == None:
            #At the error computation we have X on the left node and Round(X) on the right node.
            #Each node comes with an error, affine or symbolic.
            if self.is_error_computation:
                self.affine_error=self.leftoperand.affine_error
                self.symbolic_error=self.leftoperand.symbolic_error
            else:
                self.compute_error_affine_form()
                #self.compute_error_symbolic_form()
            if self.convolution:
                self.executeIndependent()
                self.distributionValues = self.operationDependent()
                self.distribution = self.distributionConv
                self.a = self.aConv
                self.b = self.bConv
            else:
                self.distributionValues = self.operationDependent()
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
            #val a = Interval.minAbs(rightInterval)
            #val errorMultiplier: Rational = -one / (a * a)

            total_affine_right = self.exact_affines_forms[1].\
                perform_affine_operation("+", self.rightoperand.affine_error,
                                         recursion_limit_for_pruning=recursion_limit_for_pruning_error, dReal=False)
            total_interval_right=total_affine_right.compute_interval()
            if check_zero_is_in_interval(total_interval_right):
                print("Potential division by zero!")
                exit(-1)

            min_abs_string=find_min_abs_interval(total_interval_right)
            multiplier_interval=Interval("-1.0","-1.0",True,True,digits_for_range).perform_interval_operation("/",
                                Interval(min_abs_string,min_abs_string,True,True,digits_for_range).perform_interval_operation("*",
                                Interval(min_abs_string,min_abs_string,True,True,digits_for_range)))
            multiplier_affine=AffineInstance(multiplier_interval,{})
            inv_erry=multiplier_affine.perform_affine_operation("*", self.rightoperand.affine_error)

            x_err_one_over_y=self.exact_affines_forms[0].\
                perform_affine_operation("*", inv_erry,
                                         recursion_limit_for_pruning=recursion_limit_for_pruning_error, dReal=False)
            one_over_y=self.exact_affines_forms[1].inverse()

            one_over_y_err_x=one_over_y.perform_affine_operation("*",self.leftoperand.affine_error,
                                         recursion_limit_for_pruning=recursion_limit_for_pruning_error, dReal=False)

            errx_err_one_over_y = self.leftoperand.affine_error.\
                perform_affine_operation("*", inv_erry,
                                         recursion_limit_for_pruning=recursion_limit_for_pruning_error, dReal=False)

            self.affine_error = x_err_one_over_y.perform_affine_operation("+",
                                one_over_y_err_x.perform_affine_operation("+", errx_err_one_over_y))

        elif self.operator == "*+":
            self.affine_error = self.leftoperand.affine_error.\
                perform_affine_operation("+",
                        self.exact_affines_forms[0].perform_affine_operation("+", self.leftoperand.affine_error).
                        perform_affine_operation("*",self.rightoperand.discretization.affine))
        else:
            print("Operation not supported!")
            exit(-1)

    def compute_error_symbolic_form(self):
        if self.operator == "+":
            self.symbolic_error = self.leftoperand.symbolic_error.perform_affine_operation\
                ("+",self.rightoperand.symbolic_error)
        elif self.operator == "-":
            total_affine_right = self.exact_affines_forms[3]. \
                perform_affine_operation("+", self.rightoperand.symbolic_error)
            total_affine_left = self.exact_affines_forms[2]. \
                perform_affine_operation("+", self.leftoperand.symbolic_error)

            total_interval_right = total_affine_right.compute_interval()
            total_interval_left = total_affine_left.compute_interval()
            if check_sterbenz_apply(total_interval_left, total_interval_right):
                self.do_quantize_operation=False
            self.symbolic_error = self.leftoperand.symbolic_error.perform_affine_operation \
                ("-", self.rightoperand.symbolic_error)
        elif self.operator == "*":
            #No roundoff error if one of the operands is a non - negative power of 2
            x_erry = self.exact_affines_forms[2].\
                perform_affine_operation("*", self.rightoperand.symbolic_error)
            y_errx=self.exact_affines_forms[3].\
                perform_affine_operation("*", self.leftoperand.symbolic_error)
            errx_erry=self.leftoperand.symbolic_error.\
                perform_affine_operation("*", self.rightoperand.symbolic_error)
            self.symbolic_error=x_erry.perform_affine_operation("+",
                              y_errx.perform_affine_operation("+", errx_erry))
        elif self.operator == "/":
            # - 1 / a * a
            total_affine_right = self.exact_affines_forms[3]. \
                perform_affine_operation("+", self.rightoperand.symbolic_error)

            total_interval_right = total_affine_right.compute_interval()
            if check_zero_is_in_interval(total_interval_right):
                print("Potential division by zero!")
                exit(-1)

            min_abs_string = find_min_abs_interval(total_interval_right)
            multiplier_interval = Interval("-1.0", "-1.0", True, True, digits_for_range).perform_interval_operation("/",
                                     Interval(min_abs_string,min_abs_string,True,True,digits_for_range).perform_interval_operation("*",
                                         Interval(min_abs_string,min_abs_string,True,True,digits_for_range)))

            multiplier_expression = SymbolicAffineManager.from_Interval_to_Expression(multiplier_interval)
            multiplier_symbolic = SymbolicAffineInstance(multiplier_expression,{},{})
            inv_erry = multiplier_symbolic.perform_affine_operation("*", self.rightoperand.symbolic_error)
            x_err_one_over_y = self.exact_affines_forms[2]. \
                perform_affine_operation("*", inv_erry)

            one_over_y = self.exact_affines_forms[3].inverse()
            one_over_y_err_x = one_over_y.perform_affine_operation("*", self.leftoperand.symbolic_error)

            errx_err_one_over_y = self.leftoperand.symbolic_error. \
                perform_affine_operation("*", inv_erry)

            self.symbolic_error = x_err_one_over_y.perform_affine_operation("+",
                                    one_over_y_err_x.perform_affine_operation("+", errx_err_one_over_y))
        elif self.operator == "*+":
            if self.leftoperand.do_quantize_operation:
                exponent=SymbolicAffineManager.precise_create_exp_for_Gelpia(self.exact_affines_forms[2], self.leftoperand.symbolic_error)
                self.symbolic_error = self.leftoperand.symbolic_error.\
                    perform_affine_operation("+", exponent.perform_affine_operation("*",self.rightoperand.symbolic_affine))
            else:
                self.symbolic_error = self.leftoperand.symbolic_error
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
        self.operation = operation
        self.sampleInit = True
        self.a = self.distribution.range_()[0]
        self.b = self.distribution.range_()[-1]
        self.discretization = None
        self.approximating_pair = None
        self.affine_error = None
        self.do_quantize_operation = True
        self.symbolic_error = None
        self.symbolic_affine = None
        self.get_discretization()
        self.bounding_pair = model.BoundingPair()
        self.bounding_pair.instantiate_from_distribution(self.distribution)

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
        if self.discretization is None and self.affine_error is None:
            self.discretization = self.distribution.get_discretization()
            self.affine_error = self.distribution.affine_error
            self.symbolic_error = self.distribution.symbolic_error
            self.symbolic_affine=self.distribution.symbolic_affine
        return self.discretization


# Class implementing arithmetic between either (i) two BoundingPair objects or
# (ii) a BoundingPair object and a distribution object
class BoundingPairOperation:
    def __init__(self, operation, left_operand, right_operand):
        self.operation = operation
        self.left_operand = left_operand
        self.right_operand = right_operand
        self.output = None
        if not (self.left_operand.bounding_pair.is_exact or self.right_operand.bounding_pair.is_exact):
            if self.left_operand.bounding_pair.n != self.right_operand.bounding_pair.n:
                raise ValueError("Left and right operand must be approximating pairs of the same length.")
            else:
                self.n = self.left_operand.bounding_pair.n
        elif self.right_operand.bounding_pair.is_exact:
            self.n = self.left_operand.bounding_pair.n
        elif self.left_operand.bounding_pair.is_exact:
            self.n = self.right_operand.bounding_pair.n
        else:
            self.n = 0
        # The error_bound is only used for divisions by a RV whose range includes 0
        self.error_bound = 0.0

    def perform_operation(self):
        if self.operation == "+":
            if not (self.left_operand.bounding_pair.is_exact or self.right_operand.bounding_pair.is_exact):
                self._perform_AP_Addition(self.left_operand.bounding_pair, self.right_operand.bounding_pair)
            elif self.right_operand.bounding_pair.is_exact:
                self._perform_mixed_addition("right")
            elif self.left_operand.bounding_pair.is_exact:
                self._perform_mixed_addition("left")
        elif self.operation == "-":
            if not (self.left_operand.bounding_pair.is_exact or self.right_operand.bounding_pair.is_exact):
                self._perform_AP_Subtraction(self.left_operand.bounding_pair, self.right_operand.bounding_pair)
            elif self.right_operand.bounding_pair.is_exact:
                self._perform_mixed_subtraction("right")
            elif self.left_operand.bounding_pair.is_exact:
                self._perform_mixed_subtraction("left")
        elif self.operation == "*":
            if not (self.left_operand.bounding_pair.is_exact or self.right_operand.bounding_pair.is_exact):
                self._perform_AP_Multiplication(self.left_operand.bounding_pair, self.right_operand.bounding_pair)
            elif self.right_operand.bounding_pair.is_exact:
                self._perform_mixed_multiplication("right")
            elif self.left_operand.bounding_pair.is_exact:
                self._perform_mixed_multiplcation("left")
        elif self.operation == "/":
            if not (self.left_operand.bounding_pair.is_exact or self.right_operand.bounding_pair.is_exact):
                self._perform_AP_Division(self.left_operand.bounding_pair, self.right_operand.bounding_pair)
        else:
            raise ValueError("Operation must be +, - , * or /")

        left_exact = []
        right_exact = []
        operation_exact = []
        Z = self.left_operand.distribution - self.right_operand.distribution
        for i in range(0, self.n + 1):
            left_exact.append(self.left_operand.distribution.cdf(self.left_operand.bounding_pair.support[i]))
            right_exact.append(self.right_operand.distribution.cdf(self.right_operand.bounding_pair.support[i]))
            operation_exact.append(Z.cdf(self.output.support[i]))
        plt.close("all")
        matplotlib.rcParams.update({'font.size': 10})
        fig, a = plt.subplots(3)
        a[0].plot(self.left_operand.bounding_pair.support, self.left_operand.bounding_pair.lower_cdf, "r", drawstyle='steps-post')
        a[0].plot(self.left_operand.bounding_pair.support, self.left_operand.bounding_pair.upper_cdf, "g", drawstyle='steps-pre')
        a[0].plot(self.left_operand.bounding_pair.support, left_exact, "b")
        a[0].set_title("Left operand:" + self.left_operand.distribution.getName())
        a[1].plot(self.right_operand.bounding_pair.support, self.right_operand.bounding_pair.lower_cdf, "r", drawstyle='steps-post')
        a[1].plot(self.right_operand.bounding_pair.support, self.right_operand.bounding_pair.upper_cdf, "g", drawstyle='steps-pre')
        a[1].plot(self.right_operand.bounding_pair.support, right_exact, "b")
        a[1].set_title("Right operand:" + self.left_operand.distribution.getName())
        a[2].plot(self.output.support, self.output.lower_cdf, "r", drawstyle='steps-post')
        a[2].plot(self.output.support, self.output.upper_cdf, "g", drawstyle='steps-pre')
        a[2].plot(self.output.support, operation_exact, "b")
        a[2].set_title("Operation:" + self.left_operand.distribution.getName() + self.operation + self.right_operand.distribution.getName())
        plt.show()

    def _perform_AP_Addition(self, left_bp, right_bp):
        ax = left_bp.a
        bx = left_bp.b
        ay = right_bp.a
        by = right_bp.b
        # Compute range using interval arithmetic
        a = ax + ay
        b = bx + by
        r = (b - a) / self.n
        zk = []
        uzk = []
        lzk = []
        zk.append(a)
        uzk.append(0.0)
        lzk.append(0.0)
        for k in range(1, self.n):
            z = a + (k * r)
            zk.append(z)
            l = 0
            u = 0
            for i in range(1, self.n + 1):
                j = self._l_addition(left_bp.support[i], z, right_bp.support)
                l += (left_bp.lower_cdf[i] - left_bp.upper_cdf[i - 1]) * right_bp.lower_cdf[j]
                j = self._u_addition(left_bp.support[i - 1], z, right_bp.support)
                if j == 0:
                    break
                u += (left_bp.upper_cdf[i] - left_bp.lower_cdf[i - 1]) * right_bp.upper_cdf[j]
            uzk.append(min(u, 1))
            lzk.append(max(l, 0))
        zk.append(b)
        uzk.append(1.0)
        lzk.append(1.0)
        self.output = model.BoundingPair()
        self.output.instantiate_from_arrays(zk, lzk, uzk)

    def _u_addition(self, x, z, y_array):
        i = 0
        while i < self.n + 1 and x + y_array[i] < z:
            i = i + 1
        if i < self.n + 1:
            return i
        else:
            return self.n

    def _l_addition(self, x, z, y_array):
        i = self.n
        while i >= 0 and z < x + y_array[i]:
            i = i - 1
        if i >= 0:
            return i
        else:
            return 0

    def _perform_AP_Subtraction(self, left_bp, right_bp):
        ax = left_bp.a
        bx = left_bp.b
        ay = right_bp.a
        by = right_bp.b
        # Compute range using interval arithmetic
        a = ax - by
        b = bx - ay
        r = (b - a) / self.n
        zk = []
        uzk = []
        lzk = []
        zk.append(a)
        uzk.append(0.0)
        lzk.append(0.0)
        for k in range(1, self.n):
            z = a + (k * r)
            zk.append(z)
            l = 0
            u = 0
            for i in range(1, self.n + 1):
                j = self._l_subtraction(left_bp.support[i], z, right_bp.support)
                l += (left_bp.lower_cdf[i] - left_bp.upper_cdf[i - 1]) * (1 - right_bp.upper_cdf[j])
                j = self._u_subtraction(left_bp.support[i - 1], z, right_bp.support)
                u += (left_bp.upper_cdf[i] - left_bp.lower_cdf[i - 1]) * (1 - right_bp.lower_cdf[j])
            uzk.append(min(u, 1))
            lzk.append(max(l, 0))
        zk.append(b)
        uzk.append(1.0)
        lzk.append(1.0)
        self.output = model.BoundingPair()
        self.output.instantiate_from_arrays(zk, lzk, uzk)

    def _u_subtraction(self, x, z, y_array):
        i = self.n
        while i >= 0 and x - z < y_array[i]:
            i = i - 1
        if i >= 0:
            return i
        else:
            return 0

    def _l_subtraction(self, x, z, y_array):
        i = 0
        while i < self.n + 1 and y_array[i] < x - z:
            i = i + 1
        if i < self.n + 1:
            return i
        else:
            return self.n

    # {PRECONDITION: if 0 is in the range of left_operand then 0 must be a point of discontinuity of left_operand}
    def _perform_AP_Multiplication(self, left_bp, right_bp):
        ax = self.left_operand.a
        bx = self.left_operand.b
        ay = self.right_operand.a
        by = self.right_operand.b
        # Compute range using interval arithmetic
        a = min(ax * ay, ax * by, bx * ay, bx * by)
        b = max(ax * ay, ax * by, bx * ay, bx * by)
        r = (b - a) / (self.n - 1)
        zk = []
        uzk = []
        lzk = []
        zk.append(a)
        uzk.append(0.0)
        lzk.append(0.0)
        for k in range(1, self.n):
            z = a + (k * r)
            zk.append(z)
            l = 0
            u = 0
            if z >= 0:
                for i in range(1, self.n):
                    if 0 <= self.left_operand.range_array[i - 1]:
                        j = self._l_multiplication(self.left_operand.range_array[i], z,
                                                   self.right_operand.range_array)
                        l += (self.left_operand.lower_array[i] - self.left_operand.upper_array[i - 1]) * \
                             self.right_operand.lower_array[j]
                        j = self._u_multiplication(self.left_operand.range_array[i - 1], z,
                                                   self.right_operand.range_array)
                        u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i - 1]) * \
                             self.right_operand.upper_array[j]
                    elif self.left_operand.range_array[i] <= 0:
                        j = self._l_multiplication(self.left_operand.range_array[i - 1], z,
                                                   self.right_operand.range_array)
                        l += (self.left_operand.lower_array[i] - self.left_operand.upper_array[i - 1]) * \
                             (1 - self.right_operand.upper_array[j])
                        j = self._u_multiplication(self.left_operand.range_array[i], z,
                                                   self.right_operand.range_array)
                        u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i - 1]) * \
                             (1 - self.right_operand.lower_array[j])
                    else:
                        raise ValueError("0 must be a discontinuity point")
            else:
                for i in range(1, self.n):
                    if 0 <= self.left_operand.range_array[i - 1]:
                        j = self._l_multiplication(self.left_operand.range_array[i], z,
                                                   self.right_operand.range_array)
                        l += (self.left_operand.lower_array[i] - self.left_operand.upper_array[i - 1]) * \
                             self.right_operand.upper_array[j]
                        j = self._u_multiplication(self.left_operand.range_array[i - 1], z,
                                                   self.right_operand.range_array)
                        u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i - 1]) * \
                             self.right_operand.upper_array[j]
                    elif self.left_operand.range_array[i] <= 0:
                        j = self._l_multiplication(self.left_operand.range_array[i], z,
                                                   self.right_operand.range_array)
                        l += (self.left_operand.lower_array[i] - self.left_operand.upper_array[i - 1]) * \
                             (1 - self.right_operand.upper_array[j])
                        j = self._u_multiplication(self.left_operand.range_array[i - 1], z,
                                                   self.right_operand.range_array)
                        u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i - 1]) * \
                             (1 - self.right_operand.lower_array[j])
                    else:
                        raise ValueError("0 must be a discontinuity point")
            uzk.append(min(u, 1))
            lzk.append(max(l, 0))
        self.output = model.BoundingPair()
        self.output.instantiate_from_arrays(zk, lzk, uzk)

    def _u_multiplication(self, x, z, y_array):
        if x >= 0:
            i = self.n - 1
            while i >= 0 and z / x <= y_array[i]:
                i = i - 1
            if i >= 0:
                return i
            else:
                return 0
        else:
            i = 0
            while i < self.n and y_array[i] < z / x:
                i = i + 1
            if i < self.n:
                return i
            else:
                return self.n - 1

    def _l_multiplication(self, x, z, y_array):
        if x >= 0:
            i = 0
            while i < self.n and y_array[i] < z / x:
                i = i + 1
            if i < self.n:
                return i
            else:
                return self.n - 1
        else:
            i = self.n - 1
            while i >= 0 and z / x <= y_array[i]:
                i = i - 1
            if i >= 0:
                return i
            else:
                return 0

    # {INFORMAL PRECONDITION: if the range of the right operand Y contains 0, then there must exist discontinuity points
    # u,v just below and just above 0 such that the probability that Y lies between u and v is small.
    # This is because the routine is going to remove the mass in this interval}
    # {PRECONDITION: if 0 is in the range of left_operand then 0 must be a point of discontinuity of left_operand}
    def _perform_AP_Division(self, left_bp, right_bp):
        ax = self.left_operand.a
        bx = self.left_operand.b
        ay = self.right_operand.a
        by = self.right_operand.b
        # Compute the upper or lower cdf of the right operand at zero and the range of values
        if ay < 0 < by:
            i = 0
            while self.right_operand.range_array[i] < 0:
                i += 1
            y_plus_0 = self.right_operand.upper_array[i - 1]
            y_minus_0 = self.right_operand.lower_array[i - 1]
            u = self.right_operand.range_array[i - 1]
            if self.right_operand.range_array[i] == 0:
                imax = i + 1
            else:
                imax = i
            v = self.right_operand.range_array[imax]
            a = min(ax / u, ax / v, bx / u, bx / v)
            b = max(ax / u, ax / v, bx / u, bx / v)
            self.error_bound = self.right_operand.upper_array[imax] - self.right_operand.lower_array[i-1]
        else:
            a = min(ax / ay, ax / by, bx / ay, bx / by)
            b = max(ax / ay, ax / by, bx / ay, bx / by)
            if 0 < ay:
                y_plus_0 = 0
                y_minus_0 = 0
            elif by < 1:
                y_plus_0 = 1
                y_minus_0 = 1
        r = (b - a) / (self.n - 1)
        zk = []
        uzk = []
        lzk = []
        zk.append(a)
        uzk.append(0.0)
        lzk.append(0.0)
        for k in range(1, self.n):
            z = a + (k * r)
            zk.append(z)
            l = 0
            u = 0
            if z >= 0:
                for i in range(1, self.n):
                    if 0 <= self.left_operand.range_array[i - 1]:
                        j = self._l_division(self.left_operand.range_array[i], z,
                                             self.right_operand.range_array)
                        l += (self.left_operand.lower_array[i] - self.left_operand.upper_array[i - 1]) * \
                             ((1 - self.right_operand.upper_array[j]) + y_minus_0)
                        j = self._u_division(self.left_operand.range_array[i - 1], z,
                                             self.right_operand.range_array)
                        u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i - 1]) * \
                             ((1 - self.right_operand.lower_array[j]) + y_plus_0)
                    elif self.left_operand.range_array[i] <= 0:
                        j = self._l_division(self.left_operand.range_array[i - 1], z,
                                             self.right_operand.range_array)
                        l += (self.left_operand.lower_array[i] - self.left_operand.upper_array[i - 1]) * \
                             (self.right_operand.lower_array[j] + (1 - y_plus_0))
                        j = self._u_division(self.left_operand.range_array[i], z,
                                             self.right_operand.range_array)
                        u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i - 1]) * \
                             (self.right_operand.upper_array[j] + (1 - y_minus_0))
                    else:
                        raise ValueError("0 must be a discontinuity point")
            else:
                for i in range(1, self.n):
                    if 0 <= self.left_operand.range_array[i - 1]:
                        j = self._l_division(self.left_operand.range_array[i - 1], z,
                                             self.right_operand.range_array)
                        l += (self.left_operand.lower_array[i] - self.left_operand.upper_array[i - 1]) * \
                             (y_plus_0 - self.right_operand.upper_array[j])
                        j = self._u_division(self.left_operand.range_array[i], z,
                                             self.right_operand.range_array)
                        u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i - 1]) * \
                             (y_minus_0 - self.right_operand.lower_array[j])
                    elif self.left_operand.range_array[i] <= 0:
                        j = self._l_division(self.left_operand.range_array[i], z,
                                             self.right_operand.range_array)
                        l += (self.left_operand.lower_array[i] - self.left_operand.upper_array[i - 1]) * \
                             (self.right_operand.lower_array[j] - y_plus_0)
                        j = self._u_division(self.left_operand.range_array[i - 1], z,
                                             self.right_operand.range_array)
                        u += (self.left_operand.upper_array[i] - self.left_operand.lower_array[i - 1]) * \
                             (self.right_operand.upper_array[j] - y_minus_0)
                    else:
                        raise ValueError("0 must be a discontinuity point")
            uzk.append(min(u, 1))
            lzk.append(max(l, 0))
        self.output = model.BoundingPair()
        self.output.instantiate_from_arrays(zk, lzk, uzk)

    def _u_division(self, x, z, y_array):
        if x < 0:
            i = self.n - 1
            while i >= 0 and x / z <= y_array[i]:
                i = i - 1
            if i >= 0:
                return i
            else:
                return 0
        else:
            i = 0
            while i < self.n and y_array[i] < x / z:
                i = i + 1
            if i < self.n:
                return i
            else:
                return self.n - 1

    def _l_division(self, x, z, y_array):
        if x < 0:
            i = 0
            while i < self.n and y_array[i] < x / z:
                i = i + 1
            if i < self.n:
                return i
            else:
                return self.n - 1
        else:
            i = self.n - 1
            while i >= 0 and x / z <= y_array[i]:
                i = i - 1
            if i >= 0:
                return i
            else:
                return 0

    def _perform_mixed_addition(self, exact_side="left"):
        if exact_side == "left":
            ax = self.left_operand.range_()[0]
            bx = self.left_operand.range_()[1]
            ay = self.right_operand.a
            by = self.right_operand.b
        else:
            ax = self.left_operand.a
            bx = self.left_operand.b
            ay = self.right_operand.range_()[0]
            by = self.right_operand.range_()[1]
        # Compute range using interval arithmetic
        a = ax + ay
        b = bx + by
        r = (b - a) / (self.n - 1)
        zk = []
        uzk = []
        lzk = []
        zk.append(a)
        uzk.append(0.0)
        lzk.append(0.0)
        for k in range(1, self.n):
            z = a + (k * r)
            zk.append(z)
            l = 0
            u = 0
            if exact_side == "left":
                l += self.left_operand.cdf(z - by) - self.left_operand.cdf(ax)
                u = l
            else:
                l += self.right_operand.cdf(z - bx) - self.right_operand.cdf(ay)
                u = l
            for i in range(1, self.n):
                if exact_side == "left":
                    px = self.left_operand.cdf(z - self.right_operand.range_array[i - 1]) - \
                         self.left_operand.cdf(z - self.right_operand.range_array[i])
                    l += self.right_operand.lower_array[i - 1] * px
                    u += self.right_operand.upper_array[i] * px
                else:
                    py = self.right_operand.cdf(z - self.left_operand.range_array[i - 1]) - \
                         self.right_operand.cdf(z - self.left_operand.range_array[i])
                    l += self.left_operand.lower_array[i - 1] * py
                    u += self.left_operand.upper_array[i] * py
            uzk.append(min(u, 1))
            lzk.append(max(l, 0))
        self.output = model.BoundingPair()
        self.output.instantiate_from_arrays(zk, lzk, uzk)

    def _perform_mixed_subtraction(self, exact_side="left"):
        if exact_side == "left":
            ax = self.left_operand.range_()[0]
            bx = self.left_operand.range_()[1]
            ay = self.right_operand.a
            by = self.right_operand.b
        else:
            ax = self.left_operand.a
            bx = self.left_operand.b
            ay = self.right_operand.range_()[0]
            by = self.right_operand.range_()[1]
        # Compute range using interval arithmetic
        a = ax - by
        b = bx - ay
        r = (b - a) / (self.n - 1)
        zk = []
        uzk = []
        lzk = []
        zk.append(a)
        uzk.append(0.0)
        lzk.append(0.0)
        for k in range(1, self.n):
            z = a + (k * r)
            zk.append(z)
            l = 0
            u = 0
            if exact_side == "left":
                l += self.left_operand.cdf(ay + z)
                u = l
            else:
                l += self.right_operand.cdf(by) - self.right_operand.cdf(bx - z)
                u = l
            for i in range(1, self.n):
                if exact_side == "left":
                    px = self.left_operand.cdf(self.right_operand.range_array[i] + z) - \
                         self.left_operand.cdf(self.right_operand.range_array[i - 1] + z)
                    l += (1 - self.right_operand.upper_array[i]) * px
                    u += (1 - self.right_operand.lower_array[i - 1]) * px
                else:
                    py = self.right_operand.cdf(self.left_operand.range_array[i] - z) - \
                         self.right_operand.cdf(self.left_operand.range_array[i - 1] - z)
                    l += self.left_operand.lower_array[i - 1] * py
                    u += self.left_operand.upper_array[i] * py
            uzk.append(min(u, 1))
            lzk.append(max(l, 0))
        self.output = model.BoundingPair()
        self.output.instantiate_from_arrays(zk, lzk, uzk)

    def _perform_mixed_multiplication(self, exact_side="left"):
        if exact_side == "left":
            ax = self.left_operand.range_()[0]
            bx = self.left_operand.range_()[1]
            ay = self.right_operand.a
            by = self.right_operand.b
            sign_change = 0
            while (self.right_operand.range_array[sign_change]) < 0:
                sign_change += 1
        else:
            ax = self.left_operand.a
            bx = self.left_operand.b
            ay = self.right_operand.range_()[0]
            by = self.right_operand.range_()[1]
            # Find if and where the left operand changes sign
            sign_change = 0
            while (self.left_operand.range_array[sign_change]) < 0:
                sign_change += 1
        # Compute range using interval arithmetic
        a = min(ax * ay, ax * by, bx * ay, bx * by)
        b = max(ax * ay, ax * by, bx * ay, bx * by)
        r = (b - a) / (self.n - 1)
        zk = []
        uzk = []
        lzk = []
        zk.append(a)
        uzk.append(0.0)
        lzk.append(0.0)
        for k in range(1, self.n - 1):
            z = a + (k * r)
            zk.append(z)
            if z >= 0:
                l = self.right_operand.cdf(0) + (self.right_operand.cdf(z / bx) - self.right_operand.cdf(max(0, ay)))
                u = l
                if self.left_operand.range_array[sign_change] != 0:
                    l += self.left_operand.lower_array[sign_change] * \
                        (1 - self.right_operand.cdf(z / self.left_operand.range_array[sign_change]))
                    u += self.left_operand.upper_array[sign_change] * \
                        (1 - self.right_operand.cdf(z / self.left_operand.range_array[sign_change]))
                for i in range(1, self.n):
                    if 0 <= self.left_operand.range_array[i - 1]:
                        if self.left_operand.range_array[i - 1] != 0:
                            py = (self.right_operand.cdf(z / self.left_operand.range_array[i - 1]) -
                                  self.right_operand.cdf(z / self.left_operand.range_array[i]))
                        else:
                            py = 1 - self.right_operand.cdf(z / self.left_operand.range_array[i])
                        l += self.left_operand.lower_array[i - 1] * py
                        u += self.left_operand.upper_array[i] * py
                    elif self.left_operand.range_array[i] <= 0:
                        if self.left_operand.range_array[i] != 0:
                            py = (self.right_operand.cdf(z / self.left_operand.range_array[i - 1]) -
                                  self.right_operand.cdf(z / self.left_operand.range_array[i]))
                        else:
                            py = self.right_operand.cdf(z / self.left_operand.range_array[i - 1])
                        l -= self.left_operand.upper_array[i] * py
                        u -= self.left_operand.lower_array[i - 1] * py
            else:
                l = self.right_operand.cdf(0) - (self.right_operand.cdf(min(0, by)) - self.right_operand.cdf(z / bx))
                u = l
                if self.left_operand.range_array[sign_change] != 0:
                    l -= self.left_operand.upper_array[sign_change] * \
                        (self.right_operand.cdf(z / self.left_operand.range_array[sign_change]))
                    u -= self.left_operand.lower_array[sign_change] * \
                        (self.right_operand.cdf(z / self.left_operand.range_array[sign_change]))
                for i in range(1, self.n):
                    if 0 <= self.left_operand.range_array[i - 1]:
                        if self.left_operand.range_array[i] != 0 and self.left_operand.range_array[i - 1] != 0:
                            py = (self.right_operand.cdf(z / self.left_operand.range_array[i]) -
                                  self.right_operand.cdf(z / self.left_operand.range_array[i - 1]))
                            l -= self.left_operand.upper_array[i] * py
                            u -= self.left_operand.lower_array[i - 1] * py
                    elif self.left_operand.range_array[i] <= 0:
                        if self.left_operand.range_array[i] != 0 and self.left_operand.range_array[i-1] != 0:
                            py = (self.right_operand.cdf(z / self.left_operand.range_array[i]) -
                                  self.right_operand.cdf(z / self.left_operand.range_array[i - 1]))
                            l += self.left_operand.lower_array[i - 1] * py
                            u += self.left_operand.upper_array[i] * py
            uzk.append(min(u, 1))
            lzk.append(max(l, 0))
        zk.append(b)
        uzk.append(1)
        lzk.append(1)
        self.output = model.BoundingPair()
        self.output.instantiate_from_arrays(zk, lzk, uzk)

