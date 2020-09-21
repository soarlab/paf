from interval import interval, inf, imath

def perform_interval_operation(left_lower, left_upper, operator, right_lower, right_upper):
    left_int = interval([left_lower, left_upper])
    right_int = interval([right_lower, right_upper])
    res=eval("left_int"+operator+"right_int")
    return res[0].inf, res[0].sup