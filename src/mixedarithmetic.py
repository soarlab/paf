import bisect
import copy
from decimal import Decimal

import gmpy2
import numpy as np

from AffineArithmeticLibrary import AffineInstance, AffineManager
from IntervalArithmeticLibrary import Interval
from plotting import plot_operation, plot_boxing
from project_utils import round_number_down_to_digits, dec2Str, round_near, round_down, round_up, \
    round_number_up_to_digits, round_number_nearest_to_digits
from setup_utils import digits_for_range, digits_for_cdf, digits_for_input_discretization, mpfr_proxy_precision, \
    discretization_points, use_powers_of_two_spacing

'''
A PBox consists in a domain interval associated with a probability (CDF) interval
'''
class PBox:
    def __init__(self, interval, cdf_low, cdf_up):#, input_variables):
        self.interval=interval
        self.cdf_low=cdf_low
        self.cdf_up=cdf_up
        #self.input_variables=input_variables
        self.kids=set()

    #def merge_input_variables(self, pbox):
    #    self.input_variables.update(pbox.input_variables)

    def add_kid(self, kid):
        self.kids.add(kid)

    def clear_kids(self):
        self.kids.clear()

    def sum_cdfs(self, pbox):
        tmp=Interval(self.cdf_low, self.cdf_up, True, True)\
            .addition(Interval(pbox.cdf_low, pbox.cdf_up, True, True))
        return PBox(None, tmp.lower, tmp.upper)

'''
In MixedArithmetic we combine Rigorous Interval Arithmetic with Rigorous Affine Arithmetic.
'''
class MixedArithmetic:
    def __init__(self, lower, upper, intervals):
        self.intervals = intervals
        self.affine = AffineInstance(AffineManager.compute_middle_point_given_interval(lower,upper),
                                     AffineManager.compute_uncertainty_given_interval(lower,upper))
        self.lower=None
        self.upper=None
        self.include_lower=None
        self.include_upper=None
        self.get_best_bounds()

    '''
    Compute the bounds using AA and IA and take the best.
    '''
    def get_best_bounds(self):
        if Decimal(self.intervals[0].interval.lower)>Decimal(self.affine.interval.lower):
            self.lower=self.intervals[0].interval.lower
            self.include_lower=self.intervals[0].interval.include_lower
        else:
            self.lower = self.affine.interval.lower
            self.include_lower = self.affine.interval.include_lower
        if Decimal(self.intervals[-1].interval.upper)>Decimal(self.affine.interval.upper):
            self.upper=self.affine.interval.upper
            self.include_upper=self.affine.interval.include_upper
        else:
            self.upper = self.intervals[-1].interval.upper
            self.include_upper = self.intervals[-1].interval.include_upper

    @staticmethod
    def clone_MixedArith_from_Args(affine, intervals):
        tmp=MixedArithmetic("0.0", "1.0", intervals)
        tmp.intervals=intervals
        tmp.affine=affine
        tmp.get_best_bounds()
        return tmp

    @staticmethod
    def clone_from_MixedArith(mixarith):
        return MixedArithmetic.clone_MixedArith_from_Args(mixarith.affine, mixarith.interval)

def powers_of_two_spacing():
    exp_spacing=[]
    with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundToNearest, precision=mpfr_proxy_precision) as ctx:
        exponent=gmpy2.mpfr("0")
        while abs(exponent)<discretization_points:
            value=gmpy2.exp2(exponent)
            exp_spacing.insert(0, round_number_nearest_to_digits(value, digits_for_input_discretization))
            exponent=gmpy2.sub(exponent,gmpy2.mpfr("1"))
        exp_spacing.insert(0, "0.0")
    for index, value in enumerate(exp_spacing[:-1]):
        if value==exp_spacing[index+1]:
            print("Problem with digits in powers of 2")
            exit(-1)
    return exp_spacing

def powers_of_two_error(precision):
    exp_spacing=[]
    with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundToNearest, precision=mpfr_proxy_precision) as ctx:
        exponent=gmpy2.mpfr(-precision)
        counter=0
        while counter<discretization_points//2:
            value=gmpy2.exp2(exponent)
            exp_spacing.insert(0, round_number_nearest_to_digits(value, digits_for_input_discretization))
            exponent=gmpy2.sub(exponent,gmpy2.mpfr("1"))
            counter=counter+1
    exp_spacing_reverse=copy.deepcopy(exp_spacing)
    exp_spacing_reverse.reverse()
    exp_spacing_reverse=["-"+s for s in exp_spacing_reverse]
    exp_spacing=exp_spacing_reverse+["0.0"]+exp_spacing
    for index, value in enumerate(exp_spacing[:-1]):
        if value==exp_spacing[index+1]:
            print("Problem with digits in powers of 2")
            exit(-1)
    return exp_spacing

def createDSIfromDistribution(distribution, n=50):
    #np.logspace(-9, 5, base=2, num=50) spacing should be done by powers of 2
    if distribution.range_()[0]==0.0 and distribution.range_()[-1]==1.0 and use_powers_of_two_spacing:
        lin_space = powers_of_two_spacing()
    elif "FTE" in distribution.name:
        lin_space = powers_of_two_error(distribution.d.precision)
    else:
        lin_space = np.linspace(distribution.range_()[0], distribution.range_()[-1], num=n + 1, endpoint=True)

    cdf_distr=distribution.get_piecewise_cdf()
    ret_list=[]
    for i in range(0, len(lin_space)-1):
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown, precision=mpfr_proxy_precision) as ctx:
            lower=round_number_down_to_digits(gmpy2.mpfr(lin_space[i]), digits_for_input_discretization)
        with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp, precision=mpfr_proxy_precision) as ctx:
            upper=round_number_up_to_digits(gmpy2.mpfr(lin_space[i+1]), digits_for_input_discretization)
        cdf_low_bound=min(1.0, max(0.0, cdf_distr(float(lin_space[i]))))
        cdf_up_bound=min(1.0, max(0.0, cdf_distr(float(lin_space[i+1]))))
        pbox = PBox(Interval(lower, upper, True, False, digits_for_range),
                    dec2Str(round_near(cdf_low_bound, digits_for_cdf)),
                    dec2Str(round_near(cdf_up_bound, digits_for_cdf)))
        ret_list.append(pbox)
    ret_list[0].cdf_low="0.0"
    ret_list[-1].cdf_up="1.0"
    with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundDown, precision=mpfr_proxy_precision) as ctx:
        ret_list[0].interval.lower = \
            round_number_down_to_digits(gmpy2.mpfr(distribution.a_real), digits_for_input_discretization)
    with gmpy2.local_context(gmpy2.context(), round=gmpy2.RoundUp, precision=mpfr_proxy_precision) as ctx:
        ret_list[-1].interval.upper = \
            round_number_up_to_digits(gmpy2.mpfr(distribution.b_real), digits_for_input_discretization)
    ret_list[-1].interval.include_upper = True
    mixarith=MixedArithmetic(ret_list[0].interval.lower,ret_list[-1].interval.upper,ret_list)
    return mixarith


def createAffineErrorForLeaf():
    return AffineInstance(AffineManager.compute_middle_point_given_interval("0", "0"),
                                 AffineManager.compute_uncertainty_given_interval("0", "0"))

def createPairsFromBounds(values):
    ret=[]
    tmp=set(values)
    for index,val in enumerate(values):
        if (val in tmp) and (not val==Decimal("0.0")):
            ret.append([Decimal(val),False,index])
            tmp.remove(val)
    return ret

'''
It converts a set of PBox with PDFs to a Dempster-Shafer Structure
'''
def from_PDFS_PBox_to_DSI(insiders, evaluation_points):
    insiders_low = sorted(insiders, key=lambda x: Decimal(x.interval.lower), reverse=False)
    insiders_low_value = [Decimal(x.interval.lower) for x in insiders_low]
    insiders_up = sorted(insiders, key=lambda x: Decimal(x.interval.upper), reverse=False)
    insiders_up_value = [Decimal(x.interval.upper) for x in insiders_up]

    evaluation_points=sorted(evaluation_points)

    edge_cdf = []
    val_cdf_low = []
    val_cdf_up = []

    res_ub = Decimal("0")
    acc_ub = Decimal("0")
    res_lb = Decimal("0")
    acc_lb = Decimal("0")

    #res_ub = Interval("0","0", True, True, digits_for_cdf)
    #acc_ub = Interval("0","0", True, True, digits_for_cdf)
    #res_lb = Interval("0","0", True, True, digits_for_cdf)
    #acc_lb = Interval("0","0", True, True, digits_for_cdf)

    previous_index_lower=0
    previous_index_upper=0

    for ev_point in evaluation_points:
        index = bisect.bisect_right(insiders_low_value, ev_point)
        res_ub=acc_ub
        for inside in insiders_low[previous_index_lower:index]:
            if not Decimal(inside.cdf_low)==Decimal(inside.cdf_up):
                print("This is a PDFs operation. They must match!")
                exit(-1)
            #FIX: In case the lower interval is excluded we should not sum the probability.
            #On the contrary, for the lower bound it is correct using <=.

            if Decimal(inside.interval.lower) < ev_point:
                res_ub = res_ub + Decimal(inside.cdf_low)
                #res_ub = res_ub.perform_interval_operation("+", inside)
            elif Decimal(inside.interval.lower) == ev_point and inside.interval.include_lower:
                res_ub = res_ub + Decimal(inside.cdf_low)
                #res_ub = res_ub.perform_interval_operation("+", inside)
            acc_ub = acc_ub + Decimal(inside.cdf_low)
            #acc_ub=acc_ub.perform_interval_operation("+", inside)

        previous_index_lower=index

        index = bisect.bisect_right(insiders_up_value, ev_point)
        res_lb=acc_lb
        for inside in insiders_up[previous_index_upper:index]:
            if not Decimal(inside.cdf_low)==Decimal(inside.cdf_up):
                print("This is a PDFs operation. They must match!")
                exit(-1)
            if Decimal(inside.interval.upper) <= ev_point:
                res_lb=res_lb+Decimal(inside.cdf_low)
            acc_lb=acc_lb+Decimal(inside.cdf_low)
        previous_index_upper=index

        edge_cdf.append(dec2Str(ev_point))
        cdf_lower = min(Decimal("1.0"), max(Decimal("0.0"), res_lb))
        cdf_upper = min(Decimal("1.0"), max(Decimal("0.0"), res_ub))
        val_cdf_low.append(dec2Str(cdf_lower))
        val_cdf_up.append(dec2Str(cdf_upper))

    return edge_cdf, val_cdf_low, val_cdf_up

'''
It converts a set of PBox with CDFs to a Dempster-Shafer Structure
'''
def from_CDFS_PBox_to_DSI(insiders, evaluation_points):

    #First check wheter the insiders sum up to 1
    sum=Decimal("0")
    for box in insiders:
        sum=sum+Decimal(box.cdf_up)-Decimal(box.cdf_low)
    print("Check PDF from_CDFS_PBox_to_DSI: "+str(sum))

    #Assign to the pbox the probability mass associated with the interval
    for box in insiders:
        pdf_val=Decimal(box.cdf_up)-Decimal(box.cdf_low)
        box.cdf_low=dec2Str(pdf_val)
        box.cdf_up=dec2Str(pdf_val)

    return from_PDFS_PBox_to_DSI(insiders, evaluation_points)

def convertListToDecimals(my_list):
    ret_list=[]
    for element in my_list:
        ret_list.append(Decimal(element))
    return ret_list

def from_DSI_to_PBox(edges_lower, values_lower, edges_upper, values_upper):
    edges_lower=convertListToDecimals(edges_lower)
    values_lower=convertListToDecimals(values_lower)
    edges_upper=convertListToDecimals(edges_upper)
    values_upper=convertListToDecimals(values_upper)

    #plot_operation(edges_lower,values_lower,values_upper)

    ret_list=[]
    if not edges_lower==edges_upper:
        print("Lists should be identical")
        exit(-1)

    pair_values_lower=createPairsFromBounds(values_lower)
    pair_values_upper=createPairsFromBounds(values_upper)

    for pair_value_upper in pair_values_upper:
        index = bisect.bisect_left(values_lower, pair_value_upper[0])
        pair_value_upper[1] = True

        #in case we go out of bounds we take the last one
        if index==len(values_lower):
            index=index-1

        #the lower bound of the forward box is the one before the current box
        lower_index=pair_value_upper[2]
        if lower_index==0:
            ret_list.append(PBox(Interval(dec2Str(edges_lower[lower_index]), dec2Str(edges_lower[index]),
                                          True, False, digits_for_range), "", dec2Str(pair_value_upper[0])))
        else:
            lower_index=lower_index-1
            ret_list.append(PBox(Interval(dec2Str(edges_lower[lower_index]), dec2Str(edges_lower[index]),
                                          False, False, digits_for_range), "", dec2Str(pair_value_upper[0])))

    for pair_value_lower in pair_values_lower:
        index = bisect.bisect_left(values_upper, pair_value_lower[0])
        pair_value_lower[1] = True
        if index==len(values_upper):
            index=index-1
        if index==pair_value_lower[2]:
            index=index-1
            ret_list.append(PBox(Interval(dec2Str(edges_lower[index]), dec2Str(edges_lower[pair_value_lower[2]]),
                                          False, False, digits_for_range), "", dec2Str(pair_value_lower[0])))
            continue
        if index>0:
            index = index - 1
            ret_list.append(PBox(Interval(dec2Str(edges_lower[index]), dec2Str(edges_lower[pair_value_lower[2]]),
                                          False, False, digits_for_range), "", dec2Str(pair_value_lower[0])))
            continue
        ret_list.append(PBox(Interval(dec2Str(edges_lower[index]), dec2Str(edges_lower[pair_value_lower[2]]),
                                      True, False, digits_for_range), "", dec2Str(pair_value_lower[0])))
    #sort by cdf, in case they are equal sort by lower bound
    ret_list.sort(key=lambda x: (float(x.cdf_up), float(x.interval.lower)))
    prec="0.0"
    for elem in ret_list:
        if not Decimal(prec)==Decimal(elem.cdf_up):
            elem.cdf_low=prec
            prec=elem.cdf_up
        else:
            elem.cdf_low="REMOVE"
    #remove useless
    ret_list= [x for x in ret_list if not x.cdf_low=="REMOVE"]
    ret_list[-1].interval.include_upper=True

    #plot_operation(edges_lower,values_lower,values_upper)
    #plot_boxing(ret_list)

    sum=0
    for box in ret_list:
        sum=sum+Decimal(box.cdf_up)-Decimal(box.cdf_low)
    print("Check PDF from bounds to intervals: "+str(sum))
    return ret_list