from interval import interval, inf, imath
from mpmath import iv

def perform_interval_operation(left_lower, left_upper, operator, right_lower, right_upper):
    left_int = iv.mpf([left_lower, left_upper])
    right_int = iv.mpf([right_lower, right_upper])
    res=eval("left_int"+operator+"right_int")
    return eval(str(res.a))[0], eval(str(res.b))[1]