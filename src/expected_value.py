from decimal import Decimal

import gmpy2

from IntervalArithmeticLibrary import Interval, empty_interval
from project_utils import round_number_nearest_to_digits, dec2Str, round_number_up_to_digits
from setup_utils import digits_for_range, mpfr_proxy_precision

'''
The expected value of a ds structure is SUM (pbox_i*p_i)
'''
def get_expected_value_ds(ds_structure):
    res=Interval("0.0","0.0",True,True,digits_for_range)
    for box in ds_structure:
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundToNearest, precision=mpfr_proxy_precision) as ctx:
            prob=round_number_nearest_to_digits(gmpy2.sub(gmpy2.mpfr(box.cdf_up), gmpy2.mpfr(box.cdf_low)), digits_for_range)
        res=res.perform_interval_operation("+", box.interval.perform_interval_operation("*", Interval(prob,prob,True,True,digits_for_range)))
    return res

def get_expected_value_square_ds(ds_structure):
    tmp=get_expected_value_ds(ds_structure)
    return (tmp.perform_interval_operation("*",tmp))

def get_square_expected_value_ds(ds_structure):
    res = Interval("0.0", "0.0", True, True, digits_for_range)
    for box in ds_structure:
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundToNearest, precision=mpfr_proxy_precision) as ctx:
            prob = round_number_nearest_to_digits(gmpy2.sub(box.cdf_up, box.cdf_low))
        res = res + box.interval.perform_interval_operation("*", box.interval).perform_interval_operation(prob, prob, True, True, digits_for_range)
    return res

'''
This is a worst-case probability distribution. It assumes the distribution is: xi_low -> pi and xi_up -> 1-pi
'''
def compute_p_i(e_interval, xi_lower, xi_upper):
    num=e_interval.perform_interval_operation("-", Interval(xi_lower, xi_lower, True, True, digits_for_range))
    den=Interval(xi_upper, xi_upper,True,True,digits_for_range).\
        perform_interval_operation("-", Interval(xi_lower, xi_lower,True,True,digits_for_range))
    return num.perform_interval_operation("/",den)


def get_expected_value_upper_bound(p1_value, p2_value, x1_lower_interval, x1_upper_interval, x2_lower_interval, x2_upper_interval):
    result=None
    _result=None

    interval_p1 = Interval(p1_value, p1_value, True, True, digits_for_range)
    interval_p2 = Interval(p2_value, p2_value, True, True, digits_for_range)

    if Decimal(p1_value)<=Decimal(p2_value):
        first=interval_p1.\
            perform_interval_operation("*", x1_upper_interval).\
            perform_interval_operation("*", x2_upper_interval)

        p2_minus_p1=interval_p2.perform_interval_operation("-", interval_p1)
        second=p2_minus_p1.\
            perform_interval_operation("*", x1_lower_interval).\
            perform_interval_operation("*", x2_upper_interval)

        one_minus_p2=Interval("1.0","1.0",True, True, digits_for_range).\
            perform_interval_operation("-", interval_p2)

        third=one_minus_p2.\
            perform_interval_operation("*", x1_lower_interval).\
            perform_interval_operation("*", x2_lower_interval)

        result=first.perform_interval_operation("+", second).perform_interval_operation("+", third)

        return result.upper
    else:

        _first = interval_p2. \
            perform_interval_operation("*", x1_upper_interval). \
            perform_interval_operation("*", x2_upper_interval)

        p1_minus_p2 = interval_p1.perform_interval_operation("-", interval_p2)

        _second = p1_minus_p2. \
            perform_interval_operation("*", x1_upper_interval). \
            perform_interval_operation("*", x2_lower_interval)

        one_minus_p1 = Interval("1.0", "1.0", True, True, digits_for_range). \
            perform_interval_operation("-", interval_p1)

        _third = one_minus_p1. \
            perform_interval_operation("*", x1_lower_interval). \
            perform_interval_operation("*", x2_lower_interval)

        _result = _first.perform_interval_operation("+", _second).perform_interval_operation("+", _third)

        return _result.upper

'''
Because X and Round(X) are extremely correlated! It is natural to think the lower bound of the product 
corresponds to uncorrelated variables.
'''
def get_expected_value_lower_bound(x1_interval, x2_interval):
    product=x1_interval.perform_interval_operation("*", x2_interval)
    return product.lower


def get_expected_value_product(ds_structure_x, ds_structure_round):
    e_x = get_expected_value_ds(ds_structure_x) # expected value interval of f(x)
    e_round_x = get_expected_value_ds(ds_structure_round) # expected value interval of Round(f(x))

    pi_x=compute_p_i(e_x, ds_structure_x[0].interval.lower, ds_structure_x[-1].interval.upper)
    pi_round_x=compute_p_i(e_round_x, ds_structure_round[0].interval.lower, ds_structure_round[-1].interval.upper)

    lower_bound_expected_value=get_expected_value_lower_bound(e_x, e_round_x)

    x1_lower=Interval(ds_structure_x[0].interval.lower,
                      ds_structure_x[0].interval.lower, True, True, digits_for_range)
    x1_upper=Interval(ds_structure_x[-1].interval.upper,
                      ds_structure_x[-1].interval.upper, True, True, digits_for_range)
    x2_lower=Interval(ds_structure_round[0].interval.lower,
                      ds_structure_round[0].interval.lower, True, True, digits_for_range)
    x2_upper=Interval(ds_structure_round[-1].interval.upper,
                      ds_structure_round[-1].interval.upper, True, True, digits_for_range)

    uppers=[]
    for p1 in [pi_x.lower, pi_x.upper]:
        for p2 in [pi_round_x.lower, pi_round_x.upper]:
            uppers.append(get_expected_value_upper_bound(p1,p2,x1_lower,x1_upper,x2_lower,x2_upper))

    if not pi_x.intersection(pi_round_x)==empty_interval:
        p1 = p2 = dec2Str(max(Decimal(pi_x.lower), Decimal(pi_round_x.lower)))
        uppers.append(get_expected_value_upper_bound(p1, p2, x1_lower, x1_upper, x2_lower, x2_upper))

    if not pi_x.intersection(pi_round_x)==empty_interval:
        p1 = p2 = dec2Str(min(Decimal(pi_x.upper), Decimal(pi_round_x.upper)))
        uppers.append(get_expected_value_upper_bound(p1, p2, x1_lower, x1_upper, x2_lower, x2_upper))


    for value in uppers:
        my_max=gmpy2.mpfr("-inf")
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp, precision=mpfr_proxy_precision) as ctx:
            my_max=max(my_max, gmpy2.mpfr(value))
        upper_bound_expected_value=round_number_up_to_digits(my_max, digits_for_range)

    return Interval(lower_bound_expected_value,upper_bound_expected_value,True,True,digits_for_range)