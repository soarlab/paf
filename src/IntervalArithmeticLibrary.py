import gmpy2
from gmpy2 import mpfr
from project_utils import reset_default_precision, round_number_up_to_digits, round_number_down_to_digits
from setup_utils import mpfr_proxy_precision, digits_for_range


def check_intervals_digits_coincide(interval1, interval2):
    if not interval1.digits==interval2.digits:
        print("Operations between two intervals with different digits")
        exit(-1)
    return

'''
sterbenz Theorem: If x <= 2 * y and y <= 2 * x then the operation (x - y) does not produce any error
'''
def check_sterbenz_apply(interval1, interval2):
    with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown, precision=mpfr_proxy_precision) as ctx:
        two_y_left = gmpy2.mul(mpfr("2.0"), mpfr(interval2.lower))
        two_x_left = gmpy2.mul(mpfr("2.0"), mpfr(interval1.lower))
    with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp, precision=mpfr_proxy_precision) as ctx:
        y_right = mpfr(interval2.upper)
        x_right = mpfr(interval1.upper)
    if x_right <= two_y_left and y_right <= two_x_left:
        return True
    return False

'''
Check whether 0.0 is in the interval
'''
def check_zero_is_in_interval(interval):
    with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown, precision=mpfr_proxy_precision) as ctx:
        left = mpfr(interval.lower)
    with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp, precision=mpfr_proxy_precision) as ctx:
        right = mpfr(interval.upper)
    if gmpy2.is_zero(left) and interval.include_lower:
        return True
    if gmpy2.is_zero(right) and interval.include_upper:
        return True
    if left<mpfr("0.0")<right:
        return True
    return False

'''
Find the min absolute value of a interval. Note the round to zero to be safe.
'''
def find_min_abs_interval(interval):
    with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundToZero, precision=mpfr_proxy_precision) as ctx:
        left = abs(mpfr(interval.lower))
        right = abs(mpfr(interval.upper))
    if left<right:
        #Now the number is positive so we can round down
        return round_number_down_to_digits(left, digits_for_range)
    else:
        #Now the number is positive so we can round down
        return round_number_down_to_digits(right, digits_for_range)
'''
Check whether the input interval contains zero.
'''
def check_interval_is_zero(interval):
    with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown, precision=mpfr_proxy_precision) as ctx:
        left = mpfr(interval.lower)
    with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp, precision=mpfr_proxy_precision) as ctx:
        right = mpfr(interval.upper)
    if gmpy2.is_zero(left) and gmpy2.is_zero(right):
        return True
    return False

'''
Find the max absolute value of a interval. Note the round away to zero to be safe.
'''
def find_max_abs_interval(interval):
    with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundAwayZero, precision=mpfr_proxy_precision) as ctx:
        left = abs(mpfr(interval.lower))
        right = abs(mpfr(interval.upper))
    if left>right:
        #Now the number is positive so we can round up
        return round_number_up_to_digits(left, digits_for_range)
    else:
        #Now the number is positive so we can round up
        return round_number_up_to_digits(right, digits_for_range)

class Interval:
    def __init__(self, lower, upper, include_lower, include_upper, digits):
        self.lower=lower
        self.upper=upper
        self.include_lower=include_lower
        self.include_upper=include_upper
        self.digits=digits

    def addition(self, interval):
        reset_default_precision()
        res_inc_left=False
        res_inc_right=False
        if self.include_lower and interval.include_lower:
            res_inc_left = True
        if self.include_upper and interval.include_upper:
            res_inc_right = True
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown, precision=mpfr_proxy_precision) as ctx:
            res_left = gmpy2.add(mpfr(self.lower), mpfr(interval.lower))
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp, precision=mpfr_proxy_precision) as ctx:
            res_right = gmpy2.add(mpfr(self.upper), mpfr(interval.upper))

        if res_left <= res_right:
            res_right = round_number_up_to_digits(res_right, self.digits)
            res_left = round_number_down_to_digits(res_left, self.digits)
            return Interval(res_left, res_right, res_inc_left, res_inc_right, self.digits)
        else:
            res_right = round_number_down_to_digits(res_right, self.digits)
            res_left = round_number_up_to_digits(res_left, self.digits)
            return Interval(res_right, res_left, res_inc_right, res_inc_left, self.digits)

    def subtraction(self, interval):
        reset_default_precision()
        res_inc_left=False
        res_inc_right=False
        if self.include_lower and interval.include_lower:
            res_inc_left = True
        if self.include_upper and interval.include_upper:
            res_inc_right = True
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown, precision=mpfr_proxy_precision) as ctx:
            res_left = gmpy2.sub(mpfr(self.lower), mpfr(interval.upper))
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp, precision=mpfr_proxy_precision) as ctx:
            res_right = gmpy2.sub(mpfr(self.upper), mpfr(interval.lower))

        if res_left<=res_right:
            res_right = round_number_up_to_digits(res_right, self.digits)
            res_left = round_number_down_to_digits(res_left, self.digits)
            return Interval(res_left, res_right, res_inc_left, res_inc_right, self.digits)
        else:
            res_right = round_number_down_to_digits(res_right, self.digits)
            res_left = round_number_up_to_digits(res_left, self.digits)
            return Interval(res_right, res_left, res_inc_right, res_inc_left, self.digits)

    def check_zero_is_in_interval(self):
        zero_is_included=False
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown, precision=mpfr_proxy_precision) as ctx:
            if mpfr(self.lower) < mpfr("0.0") < mpfr(self.upper):
                zero_is_included = True
            if mpfr(self.lower) == mpfr("0.0") and self.include_lower:
                zero_is_included = True
            if mpfr(self.upper) == mpfr("0.0") and self.include_upper:
                zero_is_included = True
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp, precision=mpfr_proxy_precision) as ctx:
            if mpfr(self.lower) < mpfr("0.0") < mpfr(self.upper):
                zero_is_included = True
            if mpfr(self.lower) == mpfr("0.0") and self.include_lower:
                zero_is_included = True
            if mpfr(self.upper) == mpfr("0.0") and self.include_upper:
                zero_is_included = True
        return zero_is_included

    def multiplication(self, interval):
        reset_default_precision()
        tmp_res_left = []
        tmp_res_right = []

        zero_is_included=self.check_zero_is_in_interval() or interval.check_zero_is_in_interval()

        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown, precision=mpfr_proxy_precision) as ctx:
            tmp_res_left.append(gmpy2.mul(mpfr(self.lower), mpfr(interval.lower)))
            tmp_res_left.append(gmpy2.mul(mpfr(self.lower), mpfr(interval.upper)))
            tmp_res_left.append(gmpy2.mul(mpfr(self.upper), mpfr(interval.lower)))
            tmp_res_left.append(gmpy2.mul(mpfr(self.upper), mpfr(interval.upper)))
            min_index = [i for i, value in enumerate(tmp_res_left) if value == min(tmp_res_left)]

        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp, precision=mpfr_proxy_precision) as ctx:
            tmp_res_right.append(gmpy2.mul(mpfr(self.lower), mpfr(interval.lower)))
            tmp_res_right.append(gmpy2.mul(mpfr(self.lower), mpfr(interval.upper)))
            tmp_res_right.append(gmpy2.mul(mpfr(self.upper), mpfr(interval.lower)))
            tmp_res_right.append(gmpy2.mul(mpfr(self.upper), mpfr(interval.upper)))
            max_index = [i for i, value in enumerate(tmp_res_right) if value == max(tmp_res_right)]

        tmp_bounds = [self.include_lower * interval.include_lower, self.include_lower * interval.include_upper,
                      self.include_upper * interval.include_lower, self.include_upper * interval.include_upper]

        res_inc_left = any([tmp_bounds[index] for index in min_index])
        res_inc_right = any([tmp_bounds[index] for index in max_index])

        min_value=tmp_res_left[min_index[0]]
        max_value=tmp_res_right[max_index[0]]

        if gmpy2.is_zero(min_value) and zero_is_included:
            res_inc_left=True

        if gmpy2.is_zero(max_value) and zero_is_included:
            res_inc_right=True

        res_left = round_number_down_to_digits(min_value, self.digits)
        res_right = round_number_up_to_digits(max_value, self.digits)
        return Interval(res_left, res_right, res_inc_left, res_inc_right, self.digits)

    def division(self, interval):
        reset_default_precision()
        tmp_res_left = []
        tmp_res_right = []

        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown, precision=mpfr_proxy_precision) as ctx:
            new_right_lower = gmpy2.div(1.0, mpfr(interval.upper))
            new_right_upper = gmpy2.div(1.0, mpfr(interval.lower))
            tmp_res_left.append(gmpy2.mul(mpfr(self.lower), mpfr(new_right_lower)))
            tmp_res_left.append(gmpy2.mul(mpfr(self.lower), mpfr(new_right_upper)))
            tmp_res_left.append(gmpy2.mul(mpfr(self.upper), mpfr(new_right_lower)))
            tmp_res_left.append(gmpy2.mul(mpfr(self.upper), mpfr(new_right_upper)))
            min_index = [i for i, value in enumerate(tmp_res_left) if value == min(tmp_res_left)]

        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp, precision=mpfr_proxy_precision) as ctx:
            new_right_lower = gmpy2.div(1.0, mpfr(interval.upper))
            new_right_upper = gmpy2.div(1.0, mpfr(interval.lower))
            tmp_res_right.append(gmpy2.mul(mpfr(self.lower), mpfr(new_right_lower)))
            tmp_res_right.append(gmpy2.mul(mpfr(self.lower), mpfr(new_right_upper)))
            tmp_res_right.append(gmpy2.mul(mpfr(self.upper), mpfr(new_right_lower)))
            tmp_res_right.append(gmpy2.mul(mpfr(self.upper), mpfr(new_right_upper)))
            max_index = [i for i, value in enumerate(tmp_res_right) if value == max(tmp_res_right)]

        # We have to swap the boundaries here   1/[1,2]=[.5, 1]
        new_right_lower_inc = interval.include_upper
        new_right_upper_inc = interval.include_lower

        tmp_bounds = [self.include_lower * new_right_lower_inc, self.include_lower * new_right_upper_inc,
                      self.include_upper * new_right_lower_inc, self.include_upper * new_right_upper_inc]

        res_inc_left = any([tmp_bounds[index] for index in min_index])
        res_inc_right = any([tmp_bounds[index] for index in max_index])

        res_left = round_number_down_to_digits(tmp_res_left[min_index[0]], self.digits)
        res_right = round_number_up_to_digits(tmp_res_right[max_index[0]], self.digits)

        return Interval(res_left, res_right, res_inc_left, res_inc_right, self.digits)

    '''
    In the intersection method we check wheter the intersection is empty with any rounding mode.
    '''
    def intersection(self, interval):
        check_intervals_digits_coincide(self,interval)
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundToNearest, precision=mpfr_proxy_precision) as ctx:
            if mpfr(self.lower)>mpfr(interval.upper) or mpfr(self.upper)<mpfr(interval.lower):
                return empty_interval

        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown, precision=mpfr_proxy_precision) as ctx:
            if mpfr(self.lower)>mpfr(interval.upper) or mpfr(self.upper)<mpfr(interval.lower):
                return empty_interval
            if mpfr(self.lower)>mpfr(interval.lower):
                min_bound=self.lower
            else:
                min_bound=interval.lower
            #min_bound=round_number_down_to_digits(max(mpfr(self.lower), mpfr(interval.lower)), self.digits)
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp, precision=mpfr_proxy_precision) as ctx:
            if mpfr(self.lower)>mpfr(interval.upper) or mpfr(self.upper)<mpfr(interval.lower):
                return empty_interval
            if mpfr(self.upper)<mpfr(interval.upper):
                max_bound=self.upper
            else:
                max_bound=interval.upper
            #max_bound=round_number_up_to_digits(min(mpfr(self.upper),mpfr(interval.upper)), self.digits)
        return Interval(min_bound, max_bound, True, True, self.digits)

    def union(self, interval):
        check_intervals_digits_coincide(self,interval)
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown, precision=mpfr_proxy_precision) as ctx:
            if mpfr(self.lower)<mpfr(interval.lower):
                min = self.lower
                include_min=self.include_lower
            elif mpfr(self.lower)>mpfr(interval.lower):
                min = interval.lower
                include_min = interval.include_lower
            else:
                min = interval.lower
                include_min = interval.include_lower or self.include_lower

        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp, precision=mpfr_proxy_precision) as ctx:
            if mpfr(self.upper)<mpfr(interval.upper):
                max = interval.upper
                include_max=interval.include_upper
            elif mpfr(self.upper)>mpfr(interval.upper):
                max = self.upper
                include_max = self.include_upper
            else:
                max = self.upper
                include_max = self.include_upper or interval.include_upper
        return Interval(min,max,include_min,include_max, self.digits)

    def __eq__(self, other):
        if not isinstance(other, Interval):
            return False
        return (self.lower == other.lower and
                self.upper == other.upper and
                self.include_upper == other.include_upper and
                self.include_lower == other.include_lower and
                self.digits == other.digits)

    def perform_interval_operation(self, operator, interval):
        reset_default_precision()
        check_intervals_digits_coincide(self,interval)
        if operator=="+":
            res_interval=self.addition(interval)
        elif operator=="-":
            res_interval=self.subtraction(interval)
        elif operator == "*":
            res_interval=self.multiplication(interval)
        elif operator == "/":
            res_interval=self.division(interval)
        elif operator =="*+":
            plus_one_interval=interval.addition(Interval("1.0", "1.0", True, True, self.digits))
            res_interval=self.multiplication(plus_one_interval)
        else:
            print("Interval Operation not supported")
            exit(-1)
        reset_default_precision()
        return res_interval

empty_interval=Interval("1.0", "-1.0", False, False, 0)
