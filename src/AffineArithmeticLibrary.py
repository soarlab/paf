import copy
from decimal import Decimal
import gmpy2
from gmpy2 import mpfr
from IntervalArithmeticLibrary import Interval
from project_utils import round_number_down_to_digits, round_number_up_to_digits, round_number_nearest_to_digits
from pruning import clean_non_linearity_affine
from setup_utils import digits_for_discretization, recursion_limit_for_pruning_operation


class AffineManager:
    i=1

    def __init__(self):
        print("Affine Manager should never be instantiated")
        raise NotImplementedError

    @staticmethod
    def get_new_error_index():
        tmp=AffineManager.i
        AffineManager.i=AffineManager.i+1
        return "E"+str(tmp)

    @staticmethod
    def round_value_to_interval(str_value):
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown) as ctx:
            res_left = mpfr(str_value)
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp) as ctx:
            res_right = mpfr(str_value)
        return Interval(round_number_down_to_digits(res_left, digits_for_discretization),
                        round_number_up_to_digits(res_right, digits_for_discretization), True, True, digits_for_discretization)

    @staticmethod
    def compute_middle_point_given_interval(low, upper):
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown) as ctx:
            res_left = gmpy2.add(gmpy2.div(mpfr(upper), mpfr("2.0")),gmpy2.div(mpfr(low), mpfr("2.0")))
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp) as ctx:
            res_right = gmpy2.add(gmpy2.div(mpfr(upper), mpfr("2.0")),gmpy2.div(mpfr(low), mpfr("2.0")))
        return Interval(round_number_down_to_digits(res_left, digits_for_discretization),
                        round_number_up_to_digits(res_right, digits_for_discretization), True, True, digits_for_discretization)

    @staticmethod
    def compute_uncertainty_given_interval(low, upper):
        coefficients={}
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown) as ctx:
            res_left = gmpy2.sub(gmpy2.div(mpfr(upper), mpfr("2.0")),gmpy2.div(mpfr(low), mpfr("2.0")))
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp) as ctx:
            res_right = gmpy2.sub(gmpy2.div(mpfr(upper), mpfr("2.0")),gmpy2.div(mpfr(low), mpfr("2.0")))
        coefficients[AffineManager.get_new_error_index()]=\
            Interval(round_number_down_to_digits(res_left, digits_for_discretization),
                     round_number_up_to_digits(res_right, digits_for_discretization), True, True, digits_for_discretization)
        return coefficients

class AffineInstance:
    def __init__(self, center, coefficients):
        self.center=center
        self.coefficients=coefficients
        self.interval=self.compute_interval()

    def add_all_coefficients_lower_abs(self):
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundAwayZero) as ctx:
            res_left=mpfr("0.0")
            for coeff in self.coefficients:
                res_left = gmpy2.add(res_left, abs(mpfr(self.coefficients[coeff].lower)))
            #now the number is positive so we can round up
            return round_number_up_to_digits(res_left, digits_for_discretization)

    def add_all_coefficients_upper_abs(self):
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundAwayZero) as ctx:
            res_right=mpfr("0.0")
            for coeff in self.coefficients:
                res_right = gmpy2.add(res_right, abs(mpfr(self.coefficients[coeff].upper)))
            # now the number is positive so we can round up
            return round_number_up_to_digits(res_right, digits_for_discretization)

    def compute_interval(self):
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown) as ctx:
            res_left=gmpy2.sub(mpfr(self.center.lower), mpfr(self.add_all_coefficients_lower_abs()))
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp) as ctx:
            res_right=gmpy2.add(mpfr(self.center.upper), mpfr(self.add_all_coefficients_upper_abs()))
        if res_left<=res_right:
            return Interval(round_number_down_to_digits(res_left, digits_for_discretization),
                            round_number_up_to_digits(res_right, digits_for_discretization), True, True, digits_for_discretization)
        else:
            return Interval(round_number_down_to_digits(res_right, digits_for_discretization),
                            round_number_up_to_digits(res_left, digits_for_discretization), True, True, digits_for_discretization)

    def addition(self, affine):
        new_center=self.center.addition(affine.center)
        new_coefficients={}
        keys=set().union(self.coefficients, affine.coefficients)
        for key in keys:
            res_int = self.coefficients.get(key,Interval("0","0",True,True,digits_for_discretization))\
                .addition(affine.coefficients.get(key,Interval("0","0",True,True,digits_for_discretization)))
            new_coefficients[key]=res_int
        return AffineInstance(new_center, new_coefficients)

    def add_constant_string(self, constant):
        constant=Interval(constant,constant,True,True, digits_for_discretization)
        new_center=self.center.addition(constant)
        return AffineInstance(new_center, copy.deepcopy(self.coefficients))

    def add_constant_interval(self, constant_interval):
        new_center=self.center.addition(constant_interval)
        return AffineInstance(new_center, copy.deepcopy(self.coefficients))

    def subtraction(self, affine):
        new_center=self.center.subtraction(affine.center)
        new_coefficients={}
        keys=set().union(self.coefficients, affine.coefficients)
        for key in keys:
            res_int = self.coefficients.get(key,Interval("0","0",True,True, digits_for_discretization))\
                .subtraction(affine.coefficients.get(key,Interval("0","0",True,True, digits_for_discretization)))
            new_coefficients[key]=res_int
        return AffineInstance(new_center, new_coefficients)

    def multiplication(self, affine, recursion_limit_for_pruning, dReal):
        new_center = self.center.multiplication(affine.center)
        new_coefficients = {}
        keys = set().union(self.coefficients, affine.coefficients)
        for key in keys:
            second_term=affine.center.multiplication(self.coefficients.get(key, Interval("0", "0", True, True, digits_for_discretization)))
            first_term=self.center.multiplication(affine.coefficients.get(key, Interval("0", "0", True, True, digits_for_discretization)))
            res=first_term.addition(second_term)
            new_coefficients[key] = res
        self_non_linear=Interval(self.add_all_coefficients_lower_abs(),self.add_all_coefficients_upper_abs(),True,True,digits_for_discretization)
        affine_non_linear=Interval(affine.add_all_coefficients_lower_abs(), affine.add_all_coefficients_upper_abs(),True,True,digits_for_discretization)
        res=self_non_linear.multiplication(affine_non_linear)
        value=max(Decimal(res.lower).copy_abs(), Decimal(res.upper).copy_abs())
        low_value, up_value=clean_non_linearity_affine(self.coefficients, affine.coefficients, value,
                                                       recursion_limit_for_pruning, dReal)
        middle_point_non_linear=AffineManager.compute_middle_point_given_interval(low_value,up_value)
        uncertainty_non_linear=AffineManager.compute_uncertainty_given_interval(low_value,up_value)
        new_center=new_center.addition(middle_point_non_linear)
        new_coefficients.update(uncertainty_non_linear)
        return AffineInstance(new_center, new_coefficients)

    def mult_constant_string(self, constant):
        const_interval=Interval(constant,constant,True,True,digits_for_discretization)
        new_center=self.center.multiplication(const_interval)
        new_coefficients = {}
        for key in self.coefficients:
            new_coefficients[key]=self.coefficients[key].multiplication(const_interval)
        affine_instance=AffineInstance(new_center, new_coefficients)
        clean_affine_instance=affine_instance.clean_affine_operation()
        return clean_affine_instance

    def inverse(self):
        self_interval=self.compute_interval()
        new_coefficients=copy.deepcopy(self.coefficients)

        if Decimal(self_interval.lower)<=Decimal("0.0")<=Decimal(self_interval.upper):
            print("Division By Zero")
            exit(-1)

        with gmpy2.local_context(gmpy2.context()) as ctx:
            a=min(abs(mpfr(self_interval.lower)),abs(mpfr(self_interval.upper)))
            b=max(abs(mpfr(self_interval.lower)),abs(mpfr(self_interval.upper)))

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

        if Decimal(self_interval.lower)<Decimal("0.0"):
            shift=shift.multiplication(Interval("-1.0","-1.0",True,True, digits_for_discretization))

        #Error of the approximation with min-range
        radius=AffineManager.compute_uncertainty_given_interval(d_min, d_max)
        #####
        res=AffineInstance(self.center,new_coefficients)
        res=res.mult_constant_string(round_number_nearest_to_digits(alpha,digits_for_discretization))
        res=res.add_constant_interval(shift)
        #The err radius is not shifted or scaled by shift and alpha
        #err_radius=AffineManager.get_new_error_index()
        res.coefficients.update(radius)
        #res.coefficients[err_radius]=radius
        return res

    def division(self, affine, recursion_limit_for_pruning, dReal):
        return self.multiplication(affine.inverse(), recursion_limit_for_pruning, dReal)

    def clean_affine_operation(self):
        keys=[]
        for key in self.coefficients:
            value=self.coefficients[key]
            if Decimal(value.lower)==Decimal("0.0") and Decimal(value.upper)==Decimal("0.0"):
                keys.append(key)
        for delete in keys:
            del self.coefficients[delete]
        self.compute_interval()
        return self

    def perform_affine_operation(self, operator, affine,
                                 recursion_limit_for_pruning=recursion_limit_for_pruning_operation, dReal=True):
        if operator=="+":
            affine_result=self.addition(affine)
        elif operator=="-":
            affine_result=self.subtraction(affine)
        elif operator == "*":
            affine_result=self.multiplication(affine, recursion_limit_for_pruning, dReal)
        elif operator == "/":
            affine_result=self.division(affine, recursion_limit_for_pruning, dReal)
        elif operator =="*+":
            plus_one=affine.add_constant_string("1.0")
            affine_result=self.multiplication(plus_one, recursion_limit_for_pruning, dReal)
        else:
            print("Interval Operation not supported")
            exit(-1)
        clean_affine=affine_result.clean_affine_operation()
        return clean_affine