import numpy as np

class PBox:
    def __init__(self, lower, upper, prob_value, inc_left_bound=True, inc_right_bound=True):
        self.lower = lower
        self.upper = upper
        self.prob = prob_value
        self.inc_left_bound=inc_left_bound
        self.inc_right_bound=inc_right_bound

def createDSIfromDistribution(distribution, n=50):
    lin_space = np.linspace(distribution.a, distribution.b, num=n)
    cdf_distr=distribution.get_piecewise_cdf()
    ret_list=[]
    for i in range(1, len(lin_space)):
        ret_list.append(PBox(lin_space[i-1], lin_space[i], abs(cdf_distr(lin_space[i])-cdf_distr(lin_space[i-1]))))
    return ret_list