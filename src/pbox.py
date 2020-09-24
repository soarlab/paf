import numpy as np
from pacal import DiscreteDistr

import model


class PBox:
    def __init__(self, lower, upper, prob_value):
        self.lower = lower
        self.upper = upper
        self.prob = prob_value
        self.is_marginal=True
        self.kids=set()

    def add_kid(self, kid):
        self.kids.add(kid)

    def clear_kids(self):
        self.kids.clear()

def createDSIfromDistribution(distribution, n=50):
    lin_space = np.linspace(distribution.range_()[0], distribution.range_()[1], num=n)
    cdf_distr=distribution.get_piecewise_cdf()
    ret_list=[]
    for i in range(1, len(lin_space)):
        ret_list.append(PBox(lin_space[i-1], lin_space[i], abs(cdf_distr(lin_space[i])-cdf_distr(lin_space[i-1]))))
    return ret_list

def createDiscreteDistrLower(name, edges, cdf_values):
    values_pdf=[]
    values_pdf.append(cdf_values[0])
    for ind in range(1,len(cdf_values)):
        values_pdf.append(cdf_values[ind]-cdf_values[ind-1])
    return DiscreteDistr(edges, values_pdf)

def createDiscreteDistrUpper(name, edges, cdf_values):
    values_pdf=[]
    values_pdf.append(cdf_values[0])
    for ind in range(1,len(cdf_values)):
        values_pdf.append(cdf_values[ind]-cdf_values[ind-1])
    return DiscreteDistr(edges, values_pdf)