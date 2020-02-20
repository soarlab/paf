import copy

from pychebfun import chebfun

from project_utils import MyFunDistr
from typical_error_model import TypicalErrorModel

from pacal.segments import PiecewiseDistribution, Segment
from pacal.utils import wrap_pdf

import numpy as np

class TypErrInterp(object):
    def __init__(self, lower, upper, interp_points):
        self.lower=lower
        self.upper=upper
        self.interp_points=interp_points
        self.name="TypErrorInter["+str(lower)+","+str(upper)+"]"
        self.interp_err_typ=chebfun(self.getTypical, domain=[self.lower, self.upper], N=self.interp_points)

    def getTypical(self, x):
        if isinstance(x, float) or isinstance(x, int) or len(x) == 1:
            if abs(x) <= 0.5:
                return 0.75
            else:
                return 0.5 * ((1.0 / x) - 1.0) + 0.25 * (((1.0 / x) - 1.0) ** 2)
        else:
            res = np.zeros(len(x))
            for index, ti in enumerate(x):
                if abs(ti) <= 0.5:
                    res[index] = 0.75
                else:
                    res[index] = 0.5 * ((1.0 / ti) - 1.0) + 0.25 * (((1.0 / ti) - 1.0) ** 2)
            return res
        exit(-1)
        # return data representation for pickled object

    def __getstate__(self):
        tmp_dict = copy.deepcopy(self.__dict__) # get attribute dictionary
        if 'interp_err_typ' in tmp_dict:
            del tmp_dict['interp_err_typ']  # remove interp_trunc_norm entry
        return tmp_dict
        # restore object state from data representation generated
        # by __getstate__

    def __setstate__(self, dict):
        self.lower = dict["lower"]
        self.upper = dict["upper"]
        self.name = dict["name"]
        self.interp_points = dict["interp_points"]
        if 'interp_err_typ' not in dict:
            self.interp_err_typ = chebfun(self.getTypical, domain=[self.lower, self.upper], N=self.interp_points)
            dict['interp_err_typ'] = self.interp_err_typ
        self.__dict__ = dict  # make dict our attribute dictionary

    def __call__(self, t):
        return self.interp_err_typ(t)

###
# Approximate Error Model given by the "Typical Distribution"
###
class FastTypicalErrorModel(TypicalErrorModel):
    """
    An implementation of the (fast) typical error distribution with three segments
    """
    def __init__(self, input_distribution=None, precision=None, **kwargs):
        super(FastTypicalErrorModel, self).__init__(input_distribution, precision)
        self.name = "FastTypicalErrorDistribution"
        self.hidden_err_model = MyFunDistr(TypErrInterp(-1.0, 1.0, 50), breakPoints =[-1.0, 1.0])

    def init_piecewise_pdf(self):
        self.piecewise_pdf=self.hidden_err_model.get_piecewise_pdf()

    def rand_raw(self, n=None):
        return self.hidden_err_model.rand(n)

    def pdf(self, x):
        return self.hidden_err_model(x)

    def execute(self):
        return self.hidden_err_model