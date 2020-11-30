import copy
import shlex
import subprocess
from decimal import Decimal
import gmpy2
from gmpy2 import mpfr

from AffineArithmeticLibrary import AffineManager, AffineInstance
from IntervalArithmeticLibrary import Interval
from project_utils import round_number_down_to_digits, round_number_up_to_digits, round_number_nearest_to_digits
from pruning import clean_non_linearity_affine
from setup_utils import digits_for_discretization, recursion_limit_for_pruning_operation, \
    GELPHIA_exponent_function_name, path_to_gelpia_executor


def CreateSymbolicErrorForDistributions(distribution_name, lb, ub):
    var_name={distribution_name:[lb,ub]}
    return SymbolicAffineInstance(SymExpression(distribution_name), {}, var_name)

def CreateSymbolicErrorForErrors(eps_symbol):
    err_name=SymbolicAffineManager.get_new_error_index()
    coefficients={err_name:SymExpression(eps_symbol)}
    return SymbolicAffineInstance(SymExpression("0"), coefficients, {})

def CreateSymbolicZero():
    return SymbolicAffineInstance(SymExpression("0"), {}, {})


class SymExpression:
    def __init__(self, value):
        self.value=value

    def addition(self, operand):
        res = SymExpression("(" + self.value + " + " + operand.value + ")")
        return res

    def subtraction(self, operand):
        res = SymExpression("(" + self.value + " - " + operand.value + ")")
        return res

    def multiplication(self, operand):
        res = SymExpression("(" + self.value + " * " + operand.value + ")")
        return res

    def division(self, operand):
        res = SymExpression("(" + self.value + " / " + operand.value + ")")
        return res

    def abs(self):
        res=SymExpression("abs("+self.value+")")
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
    def precise_create_exp_for_Gelpia(exact, error):
        err_interval=error.compute_interval()
        middle_point=AffineManager.compute_middle_point_given_interval(err_interval.lower, err_interval.upper)
        uncertainty=AffineManager.compute_uncertainty_given_interval(err_interval.lower, err_interval.upper)
        middle_point_exp=SymbolicAffineManager.from_Interval_to_Expression(middle_point)
        new_exact=exact.add_constant_expression(middle_point_exp)
        new_uncertainty=list(uncertainty.values())[0].perform_interval_operation("*", Interval("-1","1",True,True,digits_for_discretization))
        new_uncertainty_exp=SymbolicAffineManager.from_Interval_to_Expression(new_uncertainty)
        new_exact=new_exact.add_constant_expression(new_uncertainty_exp)
        encoding="(" + GELPHIA_exponent_function_name + "(" + str(new_exact.center) + "))"
        return SymbolicAffineInstance(SymExpression(encoding),{},copy.deepcopy(exact.variables))

class SymbolicToGelpia:

    #expressions
    def __init__(self, expression, variables):
        #path_to_gelpia_exe
        self.expression=expression
        self.variables=variables

    def encode_variables(self):
        res=""
        for var in self.variables:
            res=res+var+"=["+str(self.variables[var][0])+","+str(self.variables[var][1])+"]; "
        return res

    def compute_concrete_bounds(self, debug=False):
        variables=self.encode_variables()
        result=Interval("0","0",True,True,digits_for_discretization)
        body=variables+self.expression.value
        query = path_to_gelpia_executor+' --function "'+body+'" --mode=min-max'
        if debug:
            print(query)
        proc_run = subprocess.Popen(shlex.split(query), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        out, err = proc_run.communicate() #input=str.encode(query)) #, timeout=hard_timeout)
        if not err.decode() == "":
            print(err.decode())
        res = out.decode().strip()
        for line in res.split("\n"):
            if "Minimum lower bound" in line:
                lb = line.split("Minimum lower bound")[1].strip()
            if "Maximum upper bound" in line:
                ub = line.split("Maximum upper bound")[1].strip()
        result=result.addition(Interval(lb,ub,True,True,digits_for_discretization))#"["+lb+","+ub+"]"
        return result.lower,result.upper

class SymbolicAffineInstance:
    #center is an Expression
    #coefficients is a dictionary Ei => Expression
    #variables is a dictionary "VariableName" => [lb, ub]

    def __init__(self, center, coefficients, variables):
        self.center=center
        self.coefficients=coefficients
        self.variables=variables

    def compute_interval(self):
        self_coefficients = self.add_all_coefficients_abs_exact()
        lower_expr=self.center.subtraction(self_coefficients)
        upper_expr=self.center.addition(self_coefficients)
        lower_concrete, _ = SymbolicToGelpia(lower_expr, self.variables).compute_concrete_bounds()
        _, upper_concrete = SymbolicToGelpia(upper_expr, self.variables).compute_concrete_bounds()
        return Interval(lower_concrete, upper_concrete, True, True, digits_for_discretization)

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

    def multiplication(self, sym_affine):
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

        lower_non_linear,upper_non_linear=SymbolicToGelpia(expr_non_linear, variables_non_linear).compute_concrete_bounds()

        interval_middle_point_non_linear=AffineManager.compute_middle_point_given_interval(lower_non_linear, upper_non_linear)
        dict_uncertainty_non_linear=AffineManager.compute_uncertainty_given_interval(lower_non_linear, upper_non_linear)

        key_uncertainty=list(dict_uncertainty_non_linear.keys())[0]
        interval_uncertainty_non_linear=dict_uncertainty_non_linear[key_uncertainty]

        expr_middle_point=SymbolicAffineManager.from_Interval_to_Expression(interval_middle_point_non_linear)
        expr_uncertainty=SymbolicAffineManager.from_Interval_to_Expression(interval_uncertainty_non_linear)

        dict_symbolic_non_linear={key_uncertainty:expr_uncertainty}

        new_center=new_center.addition(expr_middle_point)
        new_coefficients.update(dict_symbolic_non_linear)

        tmp_variables=copy.deepcopy(self.variables)
        tmp_variables.update(sym_affine.variables)

        return SymbolicAffineInstance(new_center, new_coefficients, tmp_variables)

    def mult_constant_string(self, constant):
        new_center=self.center.multiplication(constant)
        new_coefficients = {}
        for key in self.coefficients:
            new_coefficients[key]=self.coefficients[key].multiplication(constant)
        sym_affine_instance=SymbolicAffineInstance(new_center, new_coefficients, copy.deepcopy(self.variables))
        return sym_affine_instance

    def inverse(self):
        concrete_interval=self.compute_interval()
        new_coefficients=copy.deepcopy(self.coefficients)

        if Decimal(concrete_interval.lower)<=Decimal("0.0")<=Decimal(concrete_interval.upper):
            print("Division By Zero")
            exit(-1)

        with gmpy2.local_context(gmpy2.context()) as ctx:
            a=min(abs(mpfr(concrete_interval.lower)),abs(mpfr(concrete_interval.upper)))
            b=max(abs(mpfr(concrete_interval.lower)),abs(mpfr(concrete_interval.upper)))

        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown) as ctx:
            b_square=gmpy2.mul(b,b)
            alpha=-gmpy2.div(mpfr("1.0"),b_square)

        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp) as ctx:
            tmp=gmpy2.div(mpfr("1.0"),a)
            d_max=gmpy2.sub(tmp, gmpy2.mul(alpha,a))

        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown) as ctx:
            tmp = gmpy2.div(mpfr("1.0"), b)
            d_min = gmpy2.sub(tmp, gmpy2.mul(alpha, b))

        shift=AffineManager.compute_middle_point_given_interval(d_min, d_max)

        if Decimal(concrete_interval.lower)<Decimal("0.0"):
            shift=shift.multiplication(Interval("-1.0","-1.0",True,True, digits_for_discretization))

        #Error of the approximation with min-range
        radius=AffineManager.compute_uncertainty_given_interval(d_min, d_max)
        #####
        res=SymbolicAffineInstance(self.center, new_coefficients)
        res=res.mult_constant_string(round_number_nearest_to_digits(alpha,digits_for_discretization))
        res=res.add_constant_expression(SymbolicAffineManager.from_Interval_to_Expression(shift))
        #The err radius is not shifted or scaled by shift and alpha
        #err_radius=AffineManager.get_new_error_index()
        res.coefficients.update(radius)
        #res.coefficients[err_radius]=radius
        return res

    def add_constant_expression(self, constant):
        new_center=self.center.addition(constant)
        return SymbolicAffineInstance(new_center, copy.deepcopy(self.coefficients), copy.deepcopy(self.variables))

    def division(self, sym_affine):
        return self.multiplication(sym_affine.inverse())

    def perform_affine_operation(self, operator, affine):
        if operator=="+":
            affine_result=self.addition(affine)
        elif operator=="-":
            affine_result=self.subtraction(affine)
        elif operator == "*":
            affine_result=self.multiplication(affine)
        elif operator == "/":
            affine_result=self.division(affine)
        elif operator =="*+":
            plus_one=affine.add_constant_expression(SymExpression("1.0"))
            affine_result=self.multiplication(plus_one)
        else:
            print("Interval Operation not supported")
            exit(-1)
        return affine_result