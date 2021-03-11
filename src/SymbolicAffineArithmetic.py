import copy
import shlex
import subprocess
from decimal import Decimal
import gmpy2
from gmpy2 import mpfr

from AffineArithmeticLibrary import AffineManager
from IntervalArithmeticLibrary import Interval, check_interval_is_zero, find_min_abs_interval, find_max_abs_interval
from project_utils import round_number_nearest_to_digits, round_number_down_to_digits, round_number_up_to_digits
from setup_utils import digits_for_range, \
    GELPHIA_exponent_function_name, path_to_gelpia_executor, mpfr_proxy_precision, path_to_gelpia_constraints_executor, \
    timeout_gelpia_standard, timeout_gelpia_constraints, use_z3_when_constraints_gelpia, \
    probabilistic_handling_of_non_linearities, constraints_probabilities


def CreateSymbolicErrorForDistributions(distribution_name, lb, ub):
    var_name={distribution_name:[lb,ub]}
    return SymbolicAffineInstance(SymExpression(distribution_name), {}, var_name)

def CreateSymbolicErrorForErrors(eps_symbol, eps_value_string):
    err_name=SymbolicAffineManager.get_new_error_index()
    coefficients={err_name:SymExpression(eps_symbol)}
    variable={eps_symbol:["-"+eps_value_string, eps_value_string]}
    return SymbolicAffineInstance(SymExpression("0"), coefficients, variable)

def CreateSymbolicZero():
    return SymbolicAffineInstance(SymExpression("0"), {}, {})


class SymExpression:
    def __init__(self, value):
        self.value=value

    def addition(self, operand):
        if SymbolicAffineManager.check_expression_is_constant_zero(self) and \
           SymbolicAffineManager.check_expression_is_constant_zero(operand):
            res = SymExpression("0")
        elif SymbolicAffineManager.check_expression_is_constant_zero(self):
            res = SymExpression(operand.value)
        elif SymbolicAffineManager.check_expression_is_constant_zero(operand):
            res = SymExpression(self.value)
        else:
            res = SymExpression("(" + self.value + " + " + operand.value + ")")
        return res

    def subtraction(self, operand):
        if SymbolicAffineManager.check_expression_is_constant_zero(self) and \
           SymbolicAffineManager.check_expression_is_constant_zero(operand):
            res = SymExpression("0")
        elif SymbolicAffineManager.check_expression_is_constant_zero(self):
            res = operand.negate()
        elif SymbolicAffineManager.check_expression_is_constant_zero(operand):
            res = SymExpression(self.value)
        else:
            res = SymExpression("(" + self.value + " - " + operand.value + ")")
        return res

    def multiplication(self, operand):
        if SymbolicAffineManager.check_expression_is_constant_zero(self) or \
           SymbolicAffineManager.check_expression_is_constant_zero(operand):
            res=SymExpression("0")
        else:
            res = SymExpression("(" + self.value + " * " + operand.value + ")")
        return res

    def division(self, operand):
        if SymbolicAffineManager.check_expression_is_constant_zero(operand):
            print("Symbolic division by zero!!!!")
            exit(-1)
        elif SymbolicAffineManager.check_expression_is_constant_zero(self):
            res=SymExpression("0")
        else:
            res = SymExpression("(" + self.value + " / " + operand.value + ")")
        return res

    def abs(self):
        res=SymExpression("abs("+self.value+")")
        return res

    def inverse(self):
        if SymbolicAffineManager.check_expression_is_constant_zero(self):
            print("Symbolic division by zero!!!!")
            exit(-1)
        else:
            res = SymExpression("(1 / " + self.value + ")")
        return res

    def negate(self):
        res=SymExpression("-("+self.value+")")
        return res

    def __str__(self):
        return self.value

class SymbolicAffineManager:
    i=1

    def __init__(self):
        print("Affine Manager should never be instantiated")
        raise NotImplementedError

    @staticmethod
    def get_new_error_index():
        tmp=SymbolicAffineManager.i
        SymbolicAffineManager.i= SymbolicAffineManager.i + 1
        return "SYM_E"+str(tmp)

    @staticmethod
    def from_Interval_to_Expression(interval):
        return SymExpression("["+interval.lower+","+interval.upper+"]")

    @staticmethod
    def check_expression_is_constant_zero(expression):
        try:
            interval=Interval(expression.value,expression.value,True,True,digits_for_range)
            return check_interval_is_zero(interval)
        except:
            return False
        return False

    @staticmethod
    def compute_symbolic_uncertainty_given_interval(low, upper):
        coefficients={}
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown, precision=mpfr_proxy_precision) as ctx:
            res_left = gmpy2.sub(gmpy2.div(mpfr(upper), mpfr("2.0")),gmpy2.div(mpfr(low), mpfr("2.0")))
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp, precision=mpfr_proxy_precision) as ctx:
            res_right = gmpy2.sub(gmpy2.div(mpfr(upper), mpfr("2.0")),gmpy2.div(mpfr(low), mpfr("2.0")))
        interval=Interval(round_number_down_to_digits(res_left, digits_for_range),
                     round_number_up_to_digits(res_right, digits_for_range), True, True, digits_for_range)
        coefficients[SymbolicAffineManager.get_new_error_index()]=SymbolicAffineManager.\
            from_Interval_to_Expression(interval)
        return coefficients

    @staticmethod
    def precise_create_exp_for_Gelpia(exact, error, real_precision_constraints):

        if len(constraints_probabilities)>1:
            print("You cannot have more than one probability value in this setting")
            exit(-1)

        prob=constraints_probabilities[0]

        constraint_dict=None
        if not real_precision_constraints==None:
            constraint_dict = {}
            for constraint in real_precision_constraints:
                values = real_precision_constraints[constraint][prob]
                constraint_dict[str(constraint)] = [values[0], values[1]]

        second_order_lower, second_order_upper = \
            SymbolicToGelpia(error.center, error.variables, constraint_dict). \
                compute_concrete_bounds(debug=True, zero_output_epsilon=True)
        center_interval = Interval(second_order_lower, second_order_upper, True, True, digits_for_range)
        err_interval = error.compute_interval_error(center_interval, constraints=constraint_dict)

        new_exact=copy.deepcopy(exact)
        if not check_interval_is_zero(err_interval):
            err_expression = SymbolicAffineManager.from_Interval_to_Expression(err_interval)
            new_exact=new_exact.add_constant_expression(err_expression)
            #The exact form has only a center (no error terms)
        encoding="(" + GELPHIA_exponent_function_name + "(" + str(new_exact.center) + "))"
        return SymbolicAffineInstance(SymExpression(encoding),{},copy.deepcopy(exact.variables))

class SymbolicToGelpia:

    #expressions
    def __init__(self, expression, variables, constraints=None):
        #path_to_gelpia_exe
        self.expression=expression
        self.variables=variables
        self.constraints=constraints

    def encode_variables(self):
        res=""
        for var in self.variables:
            res=res+var+"=["+str(self.variables[var][0])+","+str(self.variables[var][1])+"]; "
        return res

    def encode_constraints(self):
        res=""
        if not self.constraints==None:
            for constraint in self.constraints:
                res=res+self.constraints[constraint][0]+"<="+constraint+"; "
                res=res+self.constraints[constraint][1]+">="+constraint+"; "
        return res

    def compute_concrete_bounds(self, debug=True, zero_output_epsilon=False):
        variables=self.encode_variables()
        constraints=self.encode_constraints()
        body=variables+str(self.expression)+"; "+constraints
        timeout_gelpia=str(timeout_gelpia_constraints) if not constraints == '' or zero_output_epsilon \
                                                        else str(timeout_gelpia_standard)
        query = (path_to_gelpia_executor if constraints=='' else path_to_gelpia_constraints_executor) \
                + (' --function "' + body +'" --mode=min-max ') \
                + (' --timeout '+ timeout_gelpia) \
                + (' -o 0' if zero_output_epsilon else '')
        if (not constraints=='') and use_z3_when_constraints_gelpia:
            query=query+" -z"
        if debug:
            print(query)
        proc_run = subprocess.Popen(shlex.split(query), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        out, err = proc_run.communicate()
        if not err.decode() == "":
            print(err.decode())
        res = out.decode().strip()
        for line in res.split("\n"):
            if "Minimum lower bound" in line:
                lb = line.split("Minimum lower bound")[1].strip()
            if "Maximum upper bound" in line:
                ub = line.split("Maximum upper bound")[1].strip()
        return lb, ub

    def compute_non_linearity(self):
        tmp_variables=copy.deepcopy(self.variables)
        memorize_eps=Interval("-1.0","1.0",True,True,digits_for_range)
        for var in tmp_variables:
            # for the moment there should be one
            if "eps" in var:
                memorize_eps=Interval(tmp_variables[var][0], tmp_variables[var][1], True, True, digits_for_range)
                tmp_variables[var]=["-1.0","1.0"]
                break
        _, coeff_upper=SymbolicToGelpia(self.expression, tmp_variables, self.constraints).compute_concrete_bounds(debug=True, zero_output_epsilon=True)
        interval=Interval("-"+coeff_upper,coeff_upper,True,True,digits_for_range).\
                            perform_interval_operation("*", memorize_eps)
        #lower_concrete=center_interval.perform_interval_operation("-", coeff_interval)
        #upper_concrete=center_interval.perform_interval_operation("+", coeff_interval)
        return interval

class SymbolicAffineInstance:
    #center is a SymExpression
    #coefficients is a dictionary "Ei" => SymExpression or "SYM_E" => SymExpression
    #variables is a dictionary "VariableName" => [lb, ub]

    def __init__(self, center, coefficients, variables):
        self.center=center
        self.coefficients=coefficients
        self.variables=variables

    def compute_interval_error(self, center_interval, constraints=None):
        tmp_variables=copy.deepcopy(self.variables)
        memorize_eps=Interval("-1.0","1.0",True,True,digits_for_range)
        for var in tmp_variables:
            # for the moment there should be one
            if "eps" in var:
                memorize_eps=Interval(tmp_variables[var][0], tmp_variables[var][1], True, True, digits_for_range)
                tmp_variables[var]=["-1.0","1.0"]
                break
        self_coefficients = self.add_all_coefficients_abs_exact()
        _, coeff_upper=SymbolicToGelpia(self_coefficients, tmp_variables, constraints).compute_concrete_bounds(debug=True, zero_output_epsilon=True)
        coeff_interval=Interval("-"+coeff_upper,coeff_upper,True,True,digits_for_range).\
                            perform_interval_operation("*", memorize_eps)
        lower_concrete=center_interval.perform_interval_operation("-", coeff_interval)
        upper_concrete=center_interval.perform_interval_operation("+", coeff_interval)
        return Interval(lower_concrete.lower, upper_concrete.upper, True, True, digits_for_range)

    def compute_interval(self):
        self_coefficients = self.add_all_coefficients_abs_exact()
        lower_expr=self.center.subtraction(self_coefficients)
        upper_expr=self.center.addition(self_coefficients)
        lower_concrete, _ = SymbolicToGelpia(lower_expr, self.variables).compute_concrete_bounds()
        _, upper_concrete = SymbolicToGelpia(upper_expr, self.variables).compute_concrete_bounds()
        return Interval(lower_concrete, upper_concrete, True, True, digits_for_range)

    def add_all_coefficients_abs_exact(self):
        res=self.add_all_coefficients_abs_over()
        if len(res)>0:
            tmp=res[0]
            for coeff in res[1:]:
                tmp=tmp.addition(coeff)
            return tmp
        else:
            return SymExpression("0")

    def add_all_coefficients_abs_over(self):
        res=[]
        for coeff in self.coefficients:
            res.append(self.coefficients[coeff].abs())
        return res

    def addition(self, sym_affine):
        new_center=self.center.addition(sym_affine.center)
        new_coefficients={}
        keys = set().union(self.coefficients, sym_affine.coefficients)
        for key in keys:
            if key in sym_affine.coefficients and key in self.coefficients:
                new_coefficients[key] = self.coefficients[key].addition(sym_affine.coefficients[key])
            elif key in sym_affine.coefficients:
                new_coefficients[key] = sym_affine.coefficients[key]
            elif key in self.coefficients:
                new_coefficients[key] = self.coefficients[key]
            else:
                print("Error in symbolic affine arithmetic")
                exit(-1)
        tmp_variables=copy.deepcopy(self.variables)
        tmp_variables.update(sym_affine.variables)
        return SymbolicAffineInstance(new_center, new_coefficients, tmp_variables)

    def subtraction(self, sym_affine):
        new_center=self.center.subtraction(sym_affine.center)
        new_coefficients={}
        keys = set().union(self.coefficients, sym_affine.coefficients)
        for key in keys:
            if key in sym_affine.coefficients and key in self.coefficients:
                new_coefficients[key] = self.coefficients[key].subtraction(sym_affine.coefficients[key])
            elif key in self.coefficients:
                new_coefficients[key] = self.coefficients[key]
            elif key in sym_affine.coefficients:
                new_coefficients[key] = sym_affine.coefficients[key].negate()
            else:
                print("Error in symbolic affine arithmetic")
                exit(-1)
        tmp_variables=copy.deepcopy(self.variables)
        tmp_variables.update(sym_affine.variables)
        return SymbolicAffineInstance(new_center, new_coefficients, tmp_variables)

    def multiplication(self, sym_affine, non_lin_constraints=None):
        new_center = self.center.multiplication(sym_affine.center)
        new_coefficients = {}
        keys = set().union(self.coefficients, sym_affine.coefficients)
        for key in keys:
            if key in sym_affine.coefficients and key in self.coefficients:
                affine_term=self.center.multiplication(sym_affine.coefficients[key])
                self_term=sym_affine.center.multiplication(self.coefficients[key])
                new_coefficients[key] = affine_term.addition(self_term)
            elif key in self.coefficients:
                new_coefficients[key] = sym_affine.center.multiplication(self.coefficients[key])
            elif key in sym_affine.coefficients:
                new_coefficients[key] = self.center.multiplication(sym_affine.coefficients[key])
            else:
                print("Error in symbolic affine arithmetic")
                exit(-1)

        self_non_linear=self.add_all_coefficients_abs_exact()
        affine_non_linear=sym_affine.add_all_coefficients_abs_exact()
        expr_non_linear=self_non_linear.multiplication(affine_non_linear)
        variables_non_linear=copy.deepcopy(self.variables)
        variables_non_linear.update(sym_affine.variables)

        if probabilistic_handling_of_non_linearities:
            if len(constraints_probabilities)>1:
                print("You cannot use the probabilistic handling of non-linearities with more than one probability constraints")
                exit(-1)
            prob = constraints_probabilities[0]
            constraint_dict = None
            if not non_lin_constraints == None:
                constraint_dict = {}
                for constraint in non_lin_constraints:
                    values = non_lin_constraints[constraint][prob]
                    constraint_dict[str(constraint)] = [values[0], values[1]]
            interval_non_linear=SymbolicToGelpia(expr_non_linear, variables_non_linear, constraint_dict).compute_non_linearity()
            #interval_non_linear=Interval("-"+upper_non_linear, upper_non_linear, True, True, digits_for_range)
        else:
            _, upper_non_linear = SymbolicToGelpia(expr_non_linear, variables_non_linear).compute_concrete_bounds()
            interval_non_linear=Interval("-"+upper_non_linear, upper_non_linear, True, True, digits_for_range)

        if not check_interval_is_zero(interval_non_linear):
            expr_non_linear = SymbolicAffineManager.from_Interval_to_Expression(interval_non_linear)
            new_center=new_center.addition(expr_non_linear)
            #interval_middle_point_non_linear=AffineManager.compute_middle_point_given_interval(lower_non_linear, upper_non_linear)
            #dict_uncertainty_non_linear=AffineManager.compute_uncertainty_given_interval(lower_non_linear, upper_non_linear)
            #key_uncertainty=list(dict_uncertainty_non_linear.keys())[0] #Note: there can be only one in dict_uncertainty_non_linear
            #interval_uncertainty_non_linear=dict_uncertainty_non_linear[key_uncertainty]
            #expr_middle_point=SymbolicAffineManager.from_Interval_to_Expression(interval_middle_point_non_linear)
            #expr_uncertainty=SymbolicAffineManager.from_Interval_to_Expression(interval_uncertainty_non_linear)
            #dict_symbolic_non_linear={key_uncertainty:expr_uncertainty}
            #new_center=new_center.addition(expr_middle_point)
            #new_coefficients.update(dict_symbolic_non_linear)

        tmp_variables=copy.deepcopy(self.variables)
        tmp_variables.update(sym_affine.variables)

        return SymbolicAffineInstance(new_center, new_coefficients, tmp_variables)

    def mult_constant_string(self, constant):
        sym_constant=SymExpression(constant)
        new_center=self.center.multiplication(sym_constant)
        new_coefficients = {}
        for key in self.coefficients:
            new_coefficients[key]=self.coefficients[key].multiplication(sym_constant)
        sym_affine_instance=SymbolicAffineInstance(new_center, new_coefficients, copy.deepcopy(self.variables))
        return sym_affine_instance

    def mult_constant_expression(self, sym_constant):
        new_center=self.center.multiplication(sym_constant)
        new_coefficients = {}
        for key in self.coefficients:
            new_coefficients[key]=self.coefficients[key].multiplication(sym_constant)
        sym_affine_instance=SymbolicAffineInstance(new_center, new_coefficients, copy.deepcopy(self.variables))
        return sym_affine_instance

    def inverse(self):
        concrete_interval=self.compute_interval()
        new_coefficients=copy.deepcopy(self.coefficients)
        new_variables=copy.deepcopy(self.variables)

        if Decimal(concrete_interval.lower)<=Decimal("0.0")<=Decimal(concrete_interval.upper):
            print("Division By Zero")
            exit(-1)

        if len(new_coefficients)==0:
            res = SymbolicAffineInstance(self.center.inverse(), new_coefficients, new_variables)
            return res

        min_a = find_min_abs_interval(concrete_interval)
        a = Interval(min_a, min_a, True, True, digits_for_range)
        max_b = find_max_abs_interval(concrete_interval)
        b = Interval(max_b, max_b, True, True, digits_for_range)
        b_square = b.perform_interval_operation("*", b)
        alpha = Interval("-1.0", "-1.0", True, True, digits_for_range).perform_interval_operation("/", b_square)
        tmp_a = Interval("1.0", "1.0", True, True, digits_for_range).perform_interval_operation("/", a)
        d_max = tmp_a.perform_interval_operation("-", alpha.perform_interval_operation("*", a))
        tmp_b = Interval("1.0", "1.0", True, True, digits_for_range).perform_interval_operation("/", b)
        d_min = tmp_b.perform_interval_operation("-", alpha.perform_interval_operation("*", b))

        shift=Interval(d_min.lower,d_max.upper,True,True,digits_for_range)

        if Decimal(concrete_interval.lower) < Decimal("0.0"):
            shift = shift.multiplication(Interval("-1.0", "-1.0", True, True, digits_for_range))

        symbolic_shift = SymbolicAffineManager.from_Interval_to_Expression(shift)

        #Error of the approximation with min-range
        #radius=AffineManager.compute_uncertainty_given_interval(d_min, d_max)
        #####
        res=SymbolicAffineInstance(self.center, new_coefficients, new_variables)

        symbolic_alpha=SymbolicAffineManager.from_Interval_to_Expression(alpha)
        res=res.mult_constant_expression(symbolic_alpha)
        res=res.add_constant_expression(symbolic_shift)
        #There is no error radius here, because the shift is symbolic
        
        return res

    def add_constant_expression(self, constant):
        new_center=self.center.addition(constant)
        return SymbolicAffineInstance(new_center, copy.deepcopy(self.coefficients), copy.deepcopy(self.variables))

    def division(self, sym_affine):
        return self.multiplication(sym_affine.inverse())

    def perform_affine_operation(self, operator, affine, eventual_constraints=None):
        if operator=="+":
            affine_result=self.addition(affine)
        elif operator=="-":
            affine_result=self.subtraction(affine)
        elif operator == "*":
            affine_result=self.multiplication(affine,eventual_constraints)
        elif operator == "/":
            affine_result=self.division(affine)
        elif operator =="*+":
            plus_one=affine.add_constant_expression(SymExpression("1.0"))
            affine_result=self.multiplication(plus_one, eventual_constraints)
        else:
            print("Interval Operation not supported")
            exit(-1)
        return affine_result