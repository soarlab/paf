import copy
from decimal import Decimal

import pacal
from pacal import UniformDistr, ConstDistr, BetaDistr
from pacal.distr import Distr
from pychebfun import chebfun
from scipy.stats import truncnorm, norm

from AffineArithmeticLibrary import AffineInstance, AffineManager
from IntervalArithmeticLibrary import Interval
from SymbolicAffineArithmetic import CreateSymbolicErrorForDistributions, SymbolicAffineInstance, SymExpression, \
    CreateSymbolicZero
from operations import BinOpDist
from mixedarithmetic import createDSIfromDistribution, MixedArithmetic, dec2Str, PBox, from_PDFS_PBox_to_DSI, \
    from_DSI_to_PBox, createAffineErrorForLeaf
from plotting import plot_operation, plot_boxing
from project_utils import MyFunDistr, normalizeDistribution
from setup_utils import global_interpolate, discretization_points, digits_for_range, sigma_for_normal_distribution
from scipy import stats


class NodeManager:
    def __init__(self):
        self.id = 0
        self.nodeDict={}

    def createNode(self, value, children=None):
        '''If the value is a scalar generate new id (no dependent operation never)'''
        if value.isScalar:
            idtmp = self.id
            self.id = self.id + 1
        elif value.name in self.nodeDict:
            idtmp = self.nodeDict[value.name]
        else:
            idtmp = self.id
            self.nodeDict[value.name]=idtmp
            self.id=self.id+1

        newNode=Node(idtmp, value, children)
        '''If there are multiple occurences of the same id it means the operation is DEPENDENT'''
        if len(newNode.id) != len(set(newNode.id)):
            newNode.value.indipendent = False
            newNode.id=list(set(newNode.id))
        return newNode

class Node:
    def __init__(self, id, value, children=None):
        self.value = value
        self.father = None
        self.id = []
        ''' The node has the id's of all his children (if any)'''
        if children:
            self.leaf = False
            self.children = children
            for child in children:
                child.father = self
                self.id = self.id + child.id
        else:
            self.leaf = True
            self.children = []
            self.id = [id]



''' 
Class used to implement the Chebfun interpolation of the truncated normal
Note the setState and getState methods. Pacal performs convolution using multiprocessing
library, so the interpolation has to be pickable.
'''
class TruncNormal(object):
    def __init__(self, lower, upper, mean, sigma, interp_points):
        self.lower=lower
        self.upper=upper
        self.mean=mean
        self.sigma=sigma
        self.interp_points=interp_points
        self.name="Stand. Norm["+str(lower)+","+str(upper)+"]"
        self.interp_trunc_norm=chebfun(self.truncatedNormal, domain=[self.lower, self.upper], N=self.interp_points)

    def truncatedNormal(self, x):
        a_trans = (self.lower - self.mean) / self.sigma
        b_trans = (self.upper - self.mean) / self.sigma
        tmp_dist = truncnorm(a_trans, b_trans, loc=self.mean, scale=self.sigma)

        if isinstance(x, float) or isinstance(x, int) or len(x) == 1:
            if x < self.lower or x > self.upper:
                return 0
            else:
                return tmp_dist.pdf(x)
        else:
            res = np.zeros(len(x))
            for index, ti in enumerate(x):
                if ti < self.lower or ti > self.upper:
                    res[index] = 0
                else:
                    res[index] = tmp_dist.pdf(ti)
            return res
        # return data representation for pickled object

    def __getstate__(self):
        tmp_dict = copy.deepcopy(self.__dict__)  # get attribute dictionary
        if 'interp_trunc_norm' in tmp_dict:
            del tmp_dict['interp_trunc_norm']  # remove interp_trunc_norm entry
        return tmp_dict
        # restore object state from data representation generated
        # by __getstate__

    def __setstate__(self, dict):
        self.lower = dict["lower"]
        self.upper = dict["upper"]
        self.mean = dict["mean"]
        self.sigma = dict["sigma"]
        self.name = dict["name"]
        self.interp_points = dict["interp_points"]
        if 'interp_trunc_norm' not in dict:
            dict['interp_trunc_norm'] = chebfun(self.truncatedNormal, domain=[self.lower, self.upper], N=self.interp_points)
        self.__dict__ = dict  # make dict our attribute dictionary

    def __call__(self, t, *args, **kwargs):
        return self.interp_trunc_norm(t)

class N(stats.rv_continuous, Distr):
    def __init__(self,name,a,b):
        super().__init__(a=a, b=b, name='TrucatedNormal')
        self.name = name
        self.sampleInit = True
        self.isScalar = False
        self.sampleSet=[]
        self.indipendent=True
        self.a = float(a)
        self.b = float(b)
        self.a_real=a
        self.b_real=b
        if self.a<0 and self.b>0:
            self.mean=0
        else:
            self.mean= self.a if self.a<self.b else self.b
        self.sigma=sigma_for_normal_distribution
        self.interpolation_points=50
        self.discretization = None
        self.affine_error = None
        self.symbolic_error = None
        self.symbolic_affine = None
        self.init_piecewise_pdf()
        self.get_discretization()

    def resetSampleInit(self):
        self.sampleInit = True

    def getName(self):
        return self.name

    def get_piecewise_pdf(self):
        """return PDF function as a PiecewiseDistribution object"""
        if self.piecewise_pdf is None:
            self.init_piecewise_pdf()
        return self.piecewise_pdf

    def init_piecewise_pdf(self):
        piecewise_pdf = PiecewiseDistribution([])
        not_norm_hidden_pdf=MyFunDistr("Truncated-Normal", TruncNormal(self.a, self.b, self.mean, self.sigma, self.interpolation_points), breakPoints=[self.a, self.b],
                   interpolated=global_interpolate)
        hidden_pdf=normalizeDistribution(not_norm_hidden_pdf, init=True)
        piecewise_pdf.addSegment(Segment(a=self.a, b=self.b,f =hidden_pdf.get_piecewise_pdf()))
        self.piecewise_pdf = piecewise_pdf

    def get_my_truncnorm(self):
        a_trans = (self.a - self.mean) / self.sigma
        b_trans = (self.b - self.mean) / self.sigma
        tmp_dist = truncnorm(a_trans, b_trans, loc=self.mean, scale=self.sigma)
        return tmp_dist

    def get_piecewise_cdf(self):
        tn=self.get_my_truncnorm()
        return tn.cdf

    def get_discretization(self):
        if self.discretization==None and self.affine_error==None and self.symbolic_error==None:
            self.discretization = createDSIfromDistribution(self, n=discretization_points)
            self.affine_error= createAffineErrorForLeaf()
            self.symbolic_error= CreateSymbolicZero()
            self.symbolic_affine = \
                CreateSymbolicErrorForDistributions(self.name, self.discretization.intervals[0].interval.lower,
                                                    self.discretization.intervals[-1].interval.upper)

        return self.discretization

    def execute(self):
        return self

    def getRepresentation(self):
        return "Standard Normal Truncated in range ["+str(self.a)+","+str(self.b)+"]"

    def getSampleSet(self,n=100000):
        #it remembers values for future operations
        if self.sampleInit:
            #a_trans= (self.a - self.mean) / self.sigma
            #b_trans= (self.b - self.mean) / self.sigma
            tmp_dist = self.get_my_truncnorm() #truncnorm(a_trans, b_trans, loc=self.mean, scale=self.sigma)
            self.sampleSet = tmp_dist.rvs(size=n)
            self.sampleInit = False
        return self.sampleSet

class B(BetaDistr):
    def __init__(self,name,a,b):
        super().__init__(alpha=float(a), beta=float(b))
        self.name = name
        self.a=self.range_()[0]
        self.b=self.range_()[-1]
        self.a_real=a
        self.b_real=b
        self.indipendent=True
        self.sampleInit = True
        self.isScalar = False
        self.sampleSet=[]
        self.discretization = None
        self.get_discretization()

    def getName(self):
        return self.name

    def execute(self):
        return self

    def get_discretization(self):
        if self.discretization==None:
            self.discretization = createDSIfromDistribution(self, n=discretization_points)
        return self.discretization

    def getRepresentation(self):
        return "Beta ["+str(self.a)+","+str(self.b)+"]"

    def resetSampleInit(self):
        self.sampleInit = True

    def getSampleSet(self,n=100000):
        #it remembers values for future operations
        if self.sampleInit:
            self.sampleSet = self.rand(n)
            self.sampleInit = False
        return self.sampleSet

''' 
Class used to implement the Chebfun interpolation of the truncated normal
Note the setState and getState methods. Pacal performs convolution using multiprocessing
library, so the interpolation has to be pickable.
'''

class CustomInterpolator(object):
    def __init__(self, interp_points, edges, values):
        self.edges=edges
        self.values=values
        self.lower=edges[0]
        self.upper=edges[-1]
        self.interp_points=interp_points
        self.name="Custom. Interpolator["+str(self.lower)+","+str(self.upper)+"]"
        self.interp_custom=chebfun(self.interpolate_method, domain=[self.lower, self.upper], N=self.interp_points)

    def compute(self, x):
        if x<self.lower or x>self.upper:
            return 0.0
        index=np.digitize(x, self.edges, right=False)
        if index - 1 >= len(self.values):
            # can happen only when x is equal to self.upper
            index = index - 1
        return abs(self.values[index - 1])

    def interpolate_method(self, x):
        if isinstance(x, float) or isinstance(x, int) or len(x) == 1:
            if x < self.lower or x > self.upper:
                return 0
            else:
                self.compute(x)
        else:
            res = np.zeros(len(x))
            for index, ti in enumerate(x):
                if ti < self.lower or ti > self.upper:
                    res[index] = 0
                else:
                    res[index] = self.compute(ti)
            return res
        exit(-1)
        # return data representation for pickled object

    def __getstate__(self):
        tmp_dict = copy.deepcopy(self.__dict__)  # get attribute dictionary
        if 'interp_custom' in tmp_dict:
            del tmp_dict['interp_custom']  # remove interp_trunc_norm entry
        return tmp_dict
        # restore object state from data representation generated
        # by __getstate__

    def __setstate__(self, dict):
        self.edges = dict["edges"]
        self.values = dict["values"]
        self.lower = dict["lower"]
        self.upper = dict["upper"]
        self.name = dict["name"]
        self.interp_points = dict["interp_points"]
        if 'interp_custom' not in dict:
            dict['interp_custom'] = chebfun(self.interpolate_method, domain=[self.lower, self.upper], N=self.interp_points)
        self.__dict__ = dict  # make dict our attribute dictionary

    def __call__(self, t, *args, **kwargs):
        return self.interp_custom(t)

def checkProbabilityDistribution(name, edges, area):
    assert edges==sorted(edges)
    assert len(area)+1==len(edges)
    if abs(sum(area)-1.0)<=0.001:
        print("Custom prob. distr. "+name+" is OK")
    else:
        print("Custom prob. distr. "+name+" does not integrate to 1")
        exit(-1)

class CustomDistr(stats.rv_continuous, Distr):
    def __init__(self,name, edges, area):
        checkProbabilityDistribution(name, edges,area)
        super().__init__(a=edges[0], b=edges[-1], name='CustomDistr')
        self.name = name
        self.sampleInit = True
        self.isScalar = False
        self.sampleSet=[]
        self.indipendent=True
        self.a = edges[0]
        self.b = edges[-1]
        self.edges=edges
        self.area=area
        self.values=self.computeValues(edges,area)
        self.discretization = []
        self.init_piecewise_pdf()
        self.get_piecewise_cdf()
        self.get_discretization()

    def get_piecewise_pdf(self):
        """return PDF function as a PiecewiseDistribution object"""
        if self.piecewise_pdf is None:
            self.init_piecewise_pdf()
        return self.piecewise_pdf

    def get_discretization(self):
        if self.discretization==None:
            self.discretization = createDSIfromDistribution(self, n=discretization_points)
        return self.discretization

    def init_piecewise_pdf(self):
        piecewise_pdf = PiecewiseDistribution([])
        for index, edge in enumerate(self.edges[:-1]):
            piecewise_pdf.addSegment(ConstSegment(edge, self.edges[index + 1], self.values[index]))
        self.piecewise_pdf = piecewise_pdf

    def testpdf(self):
        for val in [0,0.1,1, 1.1,1.2,1.3,1.4,1.5]:
            print(self._pdf(val))
        print("\n\n")

    def testcdf(self):
        for val in [0, 0.1, 1, 1.1, 1.2, 1.3, 1.4, 1.5]:
            print(self._cdf(val))
        print("\n\n")

    def testicdf(self):
        for val in [0, 0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            print(self._ppf(val))

    def computeValues(self, edges, area):
        tmp=np.zeros(len(area))
        for ind_edge, edge in enumerate(edges[:-1]):
            tmp[ind_edge]=self.area[ind_edge]/abs(edge-edges[ind_edge+1])
        return tmp

    def execute(self):
        self.get_piecewise_pdf()
        return self

    def rand_raw(self, n=None):  # None means return scalar
        return self._ppf(n)

    def __call__(self, x):
        return self.pdf(self, x)

    def range(self):
        return self.edges[0], self.edges[-1]

    def getName(self):
        return self.name

    def getRepresentation(self):
        return "Custom distribution in range ["+str(self.a)+","+str(self.b)+"]"

    def getSampleSet(self,n=100000):
        #it remembers values for future operations
        if self.sampleInit:
            self.sampleSet = self.rvs(size=n)
            self.sampleInit = False
        return self.sampleSet

    def resetSampleInit(self):
        self.sampleInit = True

    def _ppf(self, q):
        if isinstance(q, float) or isinstance(q, int):
            return self._compute_ppf(q)
        else:
            res = np.zeros(len(q))
            for index, val in enumerate(q):
                res[index] = self._compute_ppf(val)
            return res

    def _compute_ppf(self,q):
        if q<0 or q>1.0:
            print("Inverse CDF for a bad value")
            exit(-1)
        if q==0.0:
            return self.edges[0]
        if q==1.0:
            return self.edges[-1]
        #given a measure of probability returns the value of x
        cum_area = np.cumsum(self.area)
        index = np.digitize(q, cum_area, right=False)
        new_q=q-sum(self.area[0:index])
        return (new_q/self.values[index])+self.edges[index]

    def _cdf(self, x):
        if x<self.a:
            return 0.0
        elif x>self.b:
            return 1.0
        else:
            index=np.digitize(x, self.edges, right=False)
            res=sum(self.area[0:index-1])+self.values[index-1]*abs(self.edges[index-1]-x)
            return res

    def pdf(self, x):
        return self._pdf(x)

    def _pdf(self, x):
        if x<self.a or x>self.b:
            return 0.0
        index=np.digitize(x, self.edges, right=False)
        if index-1>=len(self.values):
            #can happen only when x is equal to self.b
            index=index-1
        return abs(self.values[index-1])


class U(UniformDistr):
    def __init__(self,name,a,b):
        super().__init__(a=float(a), b=float(b))
        self.name = name
        self.a=self.range_()[0]
        self.b=self.range_()[-1]
        self.a_real=a
        self.b_real=b
        self.indipendent = True
        self.sampleInit = True
        self.isScalar = False
        self.sampleSet=[]
        self.discretization = None
        self.affine_error=None
        self.symbolic_error=None
        self.symbolic_affine=None

        self.get_discretization()

    def execute(self):
        return self

    def getName(self):
        return self.name

    def get_discretization(self):
        if self.discretization==None and self.affine_error==None and self.symbolic_error==None:
            self.discretization = createDSIfromDistribution(self, n=discretization_points)
            self.affine_error= createAffineErrorForLeaf()
            self.symbolic_affine = \
                CreateSymbolicErrorForDistributions(self.name, self.discretization.intervals[0].interval.lower, self.discretization.intervals[-1].interval.upper)
            self.symbolic_error = CreateSymbolicZero()
        return self.discretization

    def getRepresentation(self):
        return "Uniform ["+str(self.a)+","+str(self.b)+"]"

    def getSampleSet(self,n=100000):
        #it remembers values for future operations
        if self.sampleInit:
            self.sampleSet = self.rand(n)
            self.sampleInit = False
        return self.sampleSet

    def resetSampleInit(self):
        self.sampleInit = True

class Number(ConstDistr):
    def __init__(self, label):
        super().__init__(c = float(label))
        self.name = label
        self.value = float(label)
        self.isScalar=True
        self.a=self.range_()[0]
        self.b=self.range_()[-1]
        self.discretization=None
        self.affine_error=None
        self.symbolic_error=None
        self.symbolic_affine=None
        self.get_discretization()

    def execute(self):
        return self

    def getRepresentation(self):
        return "Scalar("+str(self.value)+")"

    def getSampleSet(self,n=100000):
        #it remembers values for future operations
        return self.rand_raw(n)

    def resetSampleInit(self):
        pass

    def get_discretization(self):
        if self.discretization==None:
            self.discretization=self.create_discretization()
            self.affine_error= createAffineErrorForLeaf()
            self.symbolic_affine = SymbolicAffineInstance(SymExpression(self.name), {}, {})
            self.symbolic_error = CreateSymbolicZero()
        return self.discretization

    def create_discretization(self):
        return MixedArithmetic(self.name, self.name,
                               [PBox(Interval(self.name, self.name, True, True, digits_for_range), "0.0", "1.0")])

# Classes which re-implement or customize PaCal classes
import warnings

import numpy
from numpy import ceil, log, arccos, arcsin, float_power
from numpy import finfo, float32
from numpy import isposinf, isneginf, isfinite
from pacal.distr import FuncNoninjectiveDistr, AbsDistr
from pacal.integration import _integrate_with_vartransform, integrate_fejer2
from pacal.standard_distr import *
from pacal.vartransforms import VarTransform


def _shifted_arccos(x, shift):
    return arccos(x) + shift


def _shifted_arcsin(x, shift):
    return arcsin(x) + shift


def _arcsin_der(x):
    return (1 - x ** 2) ** (-0.5)


def _arccos_der(x):
    return -(1 - x ** 2) ** (-0.5)


def _strict_ceil(x):
    if x == ceil(x):
        return x + 1
    else:
        return ceil(x)


def integrate_fejer2_exp(f, log_a, log_b=None, *args, **kwargs):
    """
    Fejer2 integration from a to +oo.
    :param f: function to integrate
    :param log_a: MUST be the LOG of the lower bound of the integral to avoid instabilities
    :param log_b: MUST be the LOG of the upper bound of the integral to avoid instabilities
    """
    if isposinf(log_a):
        return 0, 0
    vt = VarTransformExp(log_a, log_U=log_b)
    return _integrate_with_vartransform(f, vt, integrate_fejer2, *args, **kwargs)


class VarTransformExp(VarTransform):
    """Exponential variable transform.
    """

    def __init__(self, log_L=0, log_U=None):
        """
        :param log_L: MUST be the LOG of the lower bound of the integral to avoid instabilities
        :param log_U: MUST be the LOG of the upper bound of the integral to avoid instabilities
        """
        if isneginf(log_L):
            # We replace 0 by the log of the smallest possible positive representable number
            self.var_min = log(finfo(float).tiny)
        else:
            self.var_min = log_L
        if log_U is None or isposinf(log_U):
            # We replace infinity by the log of the largest possible representable number
            self.var_max = log(finfo(float).max)
        else:
            self.var_max = log_U
        self.var_inf = [0]  # parameter values corresponding to infinity.  Do
        # not distinguish +oo and -oo

    def var_change(self, x):
        return log(x)

    def inv_var_change(self, y):
        return exp(y)

    def inv_var_change_deriv(self, y):
        return exp(y)


class ExpSegment(Segment):
    """
    Segment [a,b]. Only the integrate method is overridden from Segment
    """

    def __init__(self, a, b, f):
        super(ExpSegment, self).__init__(a, b, f)

    def integrate(self, log_a=None, log_b=None):
        """definite integral over interval (c, d) \cub (a, b) """
        if log_a is None or isneginf(log_a) or exp(log_a) < self.a:
            log_a = log(self.a)
        if log_b is None or log_b > log(self.b):
            log_b = log(self.b)
        i, e = integrate_fejer2_exp(self, log_a, log_b, debug_plot=False)
        return i


class PInfExpSegment(PInfSegment):
    """
    Segment = (a, inf]. Only the integrate method is overridden from PInfSegment
    """

    def __init__(self, a, f):
        super(PInfExpSegment, self).__init__(a, f)

    def integrate(self, log_a=None, log_b=None):
        if log_a is None or exp(log_a) < self.a:
            log_a = log(self.a)
        if log_b is None or isposinf(log_b):
            i, e = integrate_fejer2_exp(self.f, log_a)
        elif log_b > log_a:
            i, e = integrate_fejer2_exp(self.f, log_a, log_b)
        else:
            i, e = 0, 0
        return i


class ExpDistr(Distr):
    """Exponent of a random variable"""

    def __init__(self, d):
        """
        :param d: MUST be a PaCal distribution
        """
        self.base_distribution = d
        # Check whether the 1/t term causes a singularity at 0
        self.singularity_at_zero = self._detect_singularity()
        super(ExpDistr, self).__init__()
        self.discretization=None

    def get_discretization(self):
        if self.discretization==None:
            self.discretization = createDSIfromDistribution(self, n=discretization_points)
        return self.discretization

    def getName(self):
        return 'exp(' + self.base_distribution.getName() + ')'

    def _exp_pdf(self, x):
        return self.base_distribution.get_piecewise_pdf()(log(x)) / x

    def _exp_out_of_range(self, x):
        if log(finfo(float).max) < x:
            warnings.warn("The support of exp(" + self.getName() + ") includes numbers too large to be represented. A "
                                                                   "sub-distribution will be constructed. Check "
                                                                   "int_err() to see how bad it is.")
            return True
        if x < log(finfo(float).tiny):
            warnings.warn("The support of " + self.getName() + "includes numbers too small to be represented. A "
                                                               "sub-distribution will be constructed. Check "
                                                               "int_err() to see how bad it is.")
            return True
        return False

    def init_piecewise_pdf(self):
        # Initialize constants

        C = 0.1
        SEGMAX = 5
        self.piecewise_pdf = PiecewiseDistribution([])
        # Get the segments of base_distribution
        segs = self.base_distribution.get_piecewise_pdf().getSegments()
        if segs[0].safe_a > log(finfo(float).max):
            raise ValueError('The smallest value in the support of the input distribution is too large. Exp not '
                             'supported')
        for i in range(0, len(segs)):
            # Start with the possible problem at 0. First check if the range of the distribution morally includes 0,
            # i.e. if it includes numbers < log(finfo(float).tiny) and if there is indeed a singularity.
            if i == 0 and self.singularity_at_zero:
                # The first segment will go from 0 to the min
                # of C and exp(segs[0].safe_b)
                if log(C) < segs[0].safe_b:
                    b = log(C)
                else:
                    b = segs[0].safe_b
                # Build the distribution from 0 to b.
                # Test if the first segment of base_distribution has a right_pole
                if isinstance(segs[0], SegmentWithPole) and segs[0].left_pole:
                    self.piecewise_pdf.addSegment(ExpSegment(0, exp(b / 2), self._exp_pdf, left_pole=True))
                    self.piecewise_pdf.addSegment(SegmentWithPole(exp(b / 2), exp(b), self._exp_pdf, left_pole=False))
                else:
                    self.piecewise_pdf.addSegment(ExpSegment(0, exp(b), self._exp_pdf))
                # Add segment from b to segs[0].safe_b if necessary.
                if b < segs[0].safe_b:
                    if self._exp_out_of_range(segs[0].safe_b):
                        self.piecewise_pdf.addSegment(PInfExpSegment(exp(b), self._exp_pdf))
                    # Check if next segment is too large to integrate reliably using standard methods
                    elif segs[0].safe_b - b >= SEGMAX:
                        self.piecewise_pdf.addSegment(ExpSegment(exp(b), exp(segs[0].safe_b), self._exp_pdf))
                    else:
                        self.piecewise_pdf.addSegment(Segment(exp(b), exp(segs[0].safe_b), self._exp_pdf))
            else:
                if self._exp_out_of_range(segs[i].safe_b):
                    self.piecewise_pdf.addSegment(PInfExpSegment(exp(segs[i].safe_a), self._exp_pdf))
                    return
                # Check if segment is too large to integrate reliably using standard methods
                elif (segs[i].safe_b - segs[i].safe_a) >= SEGMAX:
                    self.piecewise_pdf.addSegment(ExpSegment(exp(segs[i].safe_a), exp(segs[i].safe_b), self._exp_pdf))
                else:
                    self.piecewise_pdf.addSegment(Segment(exp(segs[i].safe_a), exp(segs[i].safe_b), self._exp_pdf))

    def _detect_singularity(self):
        """
        :return: A boolean value. True if pdf(ln(t))/t diverges at 0, False else. Divergence here is defined by:
        pdf(ln(t)) < t ** (1 + params.pole_detection.max_pole_exponent) for a sequence of small values starting at
        the smallest positive normal number in single precision log(finfo(float).tiny)
        """
        # Test if t can ever get close to 0. For this, the pdf must be defined at ln(small t), i.e. at sufficiently
        # negative numbers. We choose log(finfo(float).eps) as the cutoff.
        cut_off = log(finfo(float).tiny)
        if self._exp_out_of_range(self.base_distribution.range_()[0]):
            # Now test for divergence.
            for i in range(50):
                u = self.base_distribution.get_piecewise_pdf()(cut_off * (2 ** i))
                v = float_power(finfo(float).eps * (2 ** i), 1 + params.pole_detection.max_pole_exponent)
                if u > v:
                    return True
        return False


class CosineDistr(FuncNoninjectiveDistr):
    """Cosine of a random variable"""

    def __init__(self, d):
        """
        :param d: MUST be a PaCal distribution
        """
        self.base_distribution = d
        self._get_intervals()
        self.pole_at_zero = False
        super(CosineDistr, self).__init__(d, fname="cos")
        self.discretization=None

    def get_discretization(self):
        if self.discretization==None:
            self.discretization = createDSIfromDistribution(self, n=discretization_points)
        return self.discretization

    def _get_intervals(self):
        """
        :return: Generates a decomposition of the real line in intervals on which the cosine function is monotone.
        On each interval the function (fs), its local inverse (f_invs), and the derivative of the local inverse
         (f_int_derivs) are recorded.
        """
        if isfinite(self.base_distribution.range_()[0]):
            a = self.base_distribution.range_()[0]
        # else we truncate the range of distribution so as to remove just a small amount of mass
        else:
            a = self.base_distribution.quantile(finfo(float32).eps)
        if isfinite(self.base_distribution.range_()[-1]):
            b = self.base_distribution.range_()[-1]
        else:
            b = self.base_distribution.quantile(1 - finfo(float32).eps)
        # Generate the intervals [k*pi, (k+1)*pi[ on which the cosine function is monotone
        self.intervals = []
        self.fs = []
        self.f_invs = []
        self.f_inv_derivs = []
        down = a
        k = _strict_ceil(a / pi - 1)
        up = a
        while up < b:
            if b < (k + 1) * pi:
                up = b
            else:
                up = (k + 1) * pi
            self.intervals.append([down, up])
            self.fs.append(cos)
            if k % 2 == 0:
                self.f_invs.append(partial(_shifted_arccos, shift=k * pi))
                self.f_inv_derivs.append(_arccos_der)
            else:
                self.f_invs.append(partial(_shifted_arcsin, shift=(2 * k + 1) * pi / 2))
                self.f_inv_derivs.append(_arcsin_der)
            k += 1
            down = up


class SineDistr(FuncNoninjectiveDistr):
    """Sine of a random variable"""

    def __init__(self, d):
        """
        :param d: MUST be a PaCal distribution
        """
        self.base_distribution = d
        self._get_intervals()
        self.pole_at_zero = False
        super(SineDistr, self).__init__(d, fname="sin")
        self.discretization=None

    def get_discretization(self):
        if self.discretization==None:
            self.discretization = createDSIfromDistribution(self, n=discretization_points)
        return self.discretization

    def _get_intervals(self):
        """
        :return: Generates a decomposition of the real line in intervals on which the sine function is monotone.
        On each interval the function (fs), its local inverse (f_invs), and the derivative of the local inverse
         (f_int_derivs) are recorded.
        """
        if isfinite(self.base_distribution.range_()[0]):
            a = self.base_distribution.range_()[0]
        # else we truncate the range of distribution so as to remove just a small amount of mass
        else:
            a = self.base_distribution.quantile(finfo(float32).eps)
        if isfinite(self.base_distribution.range_()[-1]):
            b = self.base_distribution.range_()[-1]
        else:
            b = self.base_distribution.quantile(1 - finfo(float32).eps)
        # Generate the intervals [(2k-1)*pi/2, (2k+1)*pi/2[ on which the sine function is monotone
        self.intervals = []
        self.fs = []
        self.f_invs = []
        self.f_inv_derivs = []
        down = a
        k = _strict_ceil(a / pi - 0.5)
        up = a
        while up < b:
            if b < (2 * k + 1) * pi / 2:
                up = b
            else:
                up = (2 * k + 1) * pi / 2
            self.intervals.append([down, up])
            self.fs.append(sin)
            if k % 2 == 1:
                self.f_invs.append(partial(_shifted_arccos, shift=(2 * k - 1) * pi / 2))
                self.f_inv_derivs.append(_arccos_der)
            else:
                self.f_invs.append(partial(_shifted_arcsin, shift=k * pi))
                self.f_inv_derivs.append(_arcsin_der)
            k += 1
            down = up

class AbsDistr(AbsDistr):

    def __init__(self, d, discretization, affine_error, is_error_computation, symbolic_affine, symbolic_error):
        super(AbsDistr, self).__init__(d)
        self.operand=d
        self.discretization=discretization
        self.affine_error=affine_error
        self.is_error_computation=is_error_computation
        self.symbolic_affine=symbolic_affine
        self.symbolic_error=symbolic_error
        self.elaborate_discretization()

    def get_discretization(self):
        return self.discretization

    def elaborate_discretization(self):
        evaluation_points=set()
        discretization=copy.deepcopy(self.discretization)
        for pbox in discretization.intervals:
            if Decimal(pbox.interval.lower)<Decimal("0.0")<Decimal(pbox.interval.upper):
                pbox.interval.upper=dec2Str(max(Decimal(pbox.interval.lower).copy_abs(),Decimal(pbox.interval.upper).copy_abs()))
                pbox.interval.lower="0.0"
            else:
                tmp_lower=pbox.interval.lower
                tmp_upper=pbox.interval.upper
                pbox.interval.lower = dec2Str(min(Decimal(tmp_lower).copy_abs(), Decimal(tmp_upper).copy_abs()))
                pbox.interval.upper = dec2Str(max(Decimal(tmp_lower).copy_abs(), Decimal(tmp_upper).copy_abs()))
            pdf=dec2Str(Decimal(pbox.cdf_up)-Decimal(pbox.cdf_low))
            pbox.cdf_up=pdf
            pbox.cdf_low=pdf
            evaluation_points.add(Decimal(pbox.interval.lower))
            evaluation_points.add(Decimal(pbox.interval.upper))
        evaluation_points=sorted(evaluation_points)
        if not self.is_error_computation:
            step = round(len(evaluation_points) / (discretization_points/2.0))
            evaluation_points = sorted(set(evaluation_points[::step]+[evaluation_points[-1]]))
        #here you have to do abs(affine)
        edge_cdf, val_cdf_low, val_cdf_up = from_PDFS_PBox_to_DSI(discretization.intervals, evaluation_points)
        plot_operation(edge_cdf, val_cdf_low, val_cdf_up)
        pboxes = from_DSI_to_PBox(edge_cdf, val_cdf_low, edge_cdf, val_cdf_up)
        self.discretization=MixedArithmetic.clone_MixedArith_from_Args(discretization.affine,pboxes)

def sin(d):
    """Overload the sin function."""
    if isinstance(d, Distr):
        return SineDistr(d)
    return numpy.sin(d)

def abs(d):
    """Overload the sin function."""
    if isinstance(d, Distr):
        return AbsDistr(d)
    elif isinstance(d, BinOpDist):
        d.execute()
        return AbsDistr(d.distribution, d.discretization, d.affine_error,
                        d.is_error_computation, d.symbolic_affine, d.symbolic_error)
    elif isinstance(d, UnaryOperation):
        d.execute()
        return AbsDistr(d.distribution, d.discretization)
    return numpy.abs(d)

def cos(d):
    """Overload the sin function."""
    if isinstance(d, Distr):
        return CosineDistr(d)
    return numpy.cos(d)


def exp(d):
    """Overload the exp function."""
    if isinstance(d, Distr):
        return ExpDistr(d)
    return numpy.exp(d)


def testExp():
    X = UniformDistr(0, 700)
    expX = exp(X)
    print('Error for ' + expX.getName() + ': ' + str(expX.int_error()))
    Y = UniformDistr(0, 1000)
    expY = exp(Y)
    print('Error for ' + expY.getName() + ': ' + str(expY.int_error()))
    Z = NormalDistr()
    expZ = exp(Z)
    print('Error for ' + expZ.getName() + ': ' + str(expZ.int_error()))
    U = UniformDistr(-700, 0)
    expU = exp(U)
    print('Error for ' + expU.getName() + ': ' + str(expU.int_error()))
    V = UniformDistr(-1000, 0)
    expV = exp(V)
    print('Error for ' + expV.getName() + ': ' + str(expV.int_error()))
    W = BetaDistr(0.5, 0.5)
    expW = exp(W)
    print('Error for ' + expW.getName() + ': ' + str(expW.int_error()))


def testCos():
    # this should agree with the PaCal implementation
    X = UniformDistr(0, pi)
    cosX = cos(X)
    print(cosX.summary())
    # these should show a huge improvement
    Y = UniformDistr(0, 20)
    cosY = cos(Y)
    print(cosY.summary())
    Z = NormalDistr()
    cosZ = cos(Z)
    print(cosZ.summary())


def testSin():
    # this should agree with the PaCal implementation
    X = UniformDistr(-pi / 2, pi / 2)
    sinX = sin(X)
    print(sinX.summary())
    # these should show a huge improvement
    Y = UniformDistr(0, 20)
    sinY = sin(Y)
    print(sinY.summary())
    Z = NormalDistr()
    sinZ = sin(Z)
    print(sinZ.summary())


class Operation:
    def __init__(self, leftoperand, operator, rightoperand):
        self.name = leftoperand.name + str(operator) + rightoperand.name
        self.leftoperand=leftoperand
        self.operator=operator
        self.rightoperand=rightoperand
        self.indipendent=True
        self.isScalar=False
        if leftoperand.isScalar and rightoperand.isScalar:
            self.isScalar = True

class UnaryOperation:
    def __init__(self, operand, operator):
        self.name = operator+"(" + operand.name + ")"
        self.operand=operand
        self.operator=operator
        self.indipendent=True
        self.isScalar=False
        if operand.isScalar:
            self.isScalar = True