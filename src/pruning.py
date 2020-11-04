from decimal import Decimal

import SMT_Interface
from IntervalArithmeticLibrary import Interval
from project_utils import dec2Str, linear_space_with_decimals
from setup_utils import divisions_SMT_pruning, valid_for_exit_SMT_pruning, digits_for_discretization, \
    recursion_limit_for_pruning


def clean_co_domain(pbox, smt_manager, expression_center, start_recursion_limit=0, dReal=True):
    low, sup, inc_low, inc_sup=pbox.lower,pbox.upper,pbox.include_lower,pbox.include_upper
    if Decimal(low)==Decimal(sup):
        return pbox

    codomain_intervals = linear_space_with_decimals(low, sup, inc_low, inc_sup, divisions_SMT_pruning)

    if len(codomain_intervals)==0:
        return pbox

    exists_true=0

    for interval in codomain_intervals:
        tmp_pbox = SMT_Interface.PBoxSolver(interval[0][0],interval[1][0],interval[0][1],interval[1][1])
        smt_manager.set_expression_central(expression_center, tmp_pbox)
        if not smt_manager.check(debug=False, dReal=dReal):
            if exists_true>=1:
                break
        else:
            interval[2] = True
            exists_true = exists_true+1

    if len(codomain_intervals)<divisions_SMT_pruning:
        exists_true = valid_for_exit_SMT_pruning + 1

    double_check=False
    for interval in codomain_intervals:
        if interval[2]:
            double_check=True
            break
    if not double_check:
        print("Problem with cleaning")
        exit(-1)

    for ind, interval in enumerate(codomain_intervals[:-1]):
        if not interval[2]:
            low = dec2Str(codomain_intervals[ind + 1][0][0])
            inc_low = codomain_intervals[ind + 1][0][1]
        else:
            break
    reversed_codomain=codomain_intervals[::-1]
    for ind, interval in enumerate(reversed_codomain[:-1]):
        if not interval[2]:
            sup = dec2Str(reversed_codomain[ind + 1][1][0])
            inc_sup = reversed_codomain[ind + 1][1][1]
        else:
            break
    ret_box=Interval(low,sup,inc_low,inc_sup,digits_for_discretization)
    if start_recursion_limit>recursion_limit_for_pruning:
        print("Hit the recursion limit for pruning!!\n")
        print("Low,Sup",low,sup)
    if exists_true<=valid_for_exit_SMT_pruning and start_recursion_limit<=recursion_limit_for_pruning:
        return clean_co_domain(ret_box, smt_manager, expression_center, start_recursion_limit=start_recursion_limit + 1, dReal=dReal)
    return ret_box


def clean_non_linearity_affine(left_coefficients, right_coefficients, value):
    keys = set().union(left_coefficients, right_coefficients)
    SMT_pruning = SMT_Interface.SMT_Instance()
    name_dictionary_left = {}
    name_dictionary_right = {}
    var_name = "I"
    i = 0
    for key in keys:
        SMT_pruning.add_var(key, "-1.0", "1.0")
        if key in left_coefficients:
            SMT_pruning.add_var(var_name + str(i), left_coefficients[key].lower, left_coefficients[key].upper)
            name_dictionary_left[key] = var_name + str(i)
            i = i + 1
        if key in right_coefficients:
            SMT_pruning.add_var(var_name + str(i), right_coefficients[key].lower, right_coefficients[key].upper)
            name_dictionary_right[key] = var_name + str(i)
            i = i + 1
    central = SMT_Interface.create_expression_for_multiplication(name_dictionary_left, name_dictionary_right)

    codomain=Interval(dec2Str(value.copy_negate()), dec2Str(value), True, True, digits_for_discretization)
    ret_box=clean_co_domain(codomain, SMT_pruning, central)
    return ret_box.lower, ret_box.upper