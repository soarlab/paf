import copy

import pacal
from pychebfun import chebfun
from scipy.stats import truncnorm
import numpy as np
from project_utils import MyFunDistr, normalizeDistribution
from setup_utils import global_interpolate
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
    def __init__(self, lower, upper, interp_points):
        self.lower=lower
        self.upper=upper
        self.interp_points=interp_points
        self.name="Stand. Norm["+str(lower)+","+str(upper)+"]"
        self.interp_trunc_norm=chebfun(self.truncatedNormal, domain=[self.lower, self.upper], N=self.interp_points)

    def truncatedNormal(self, x):
        tmp = pacal.NormalDistr(0, 1)
        if isinstance(x, float) or isinstance(x, int) or len(x) == 1:
            if x < self.lower or x > self.upper:
                return 0
            else:
                return tmp.get_piecewise_pdf()(x)
        else:
            res = np.zeros(len(x))
            for index, ti in enumerate(x):
                if ti < self.lower or ti > self.upper:
                    res[index] = 0
                else:
                    res[index] = tmp.get_piecewise_pdf()(ti)
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
        self.name = dict["name"]
        self.interp_points = dict["interp_points"]
        if 'interp_trunc_norm' not in dict:
            dict['interp_trunc_norm'] = chebfun(self.truncatedNormal, domain=[self.lower, self.upper], N=self.interp_points)
        self.__dict__ = dict  # make dict our attribute dictionary

    def __call__(self, t, *args, **kwargs):
        return self.interp_trunc_norm(t)

class N:
    def __init__(self,name,a,b):
        self.name = name
        self.sampleInit = True
        self.isScalar = False
        self.sampleSet=[]
        self.indipendent=True
        self.a = float(a)
        self.b = float(b)
        self.distribution = MyFunDistr(TruncNormal(self.a,self.b,50), breakPoints =[self.a, self.b], interpolated=global_interpolate)
        self.distribution.get_piecewise_pdf()
        self.distribution=normalizeDistribution(self.distribution, init=True)

    def execute(self):
        return self.distribution

    def getRepresentation(self):
        return "Standard Normal Truncated in range ["+str(self.a)+","+str(self.b)+"]"

    def getSampleSet(self,n=100000):
        #it remembers values for future operations
        if self.sampleInit:
            tmp_dist = truncnorm(self.a, self.b)
            self.sampleSet = tmp_dist.rvs(size=n)
            self.sampleInit = False
        return self.sampleSet

class B:
    def __init__(self,name,a,b):
        self.name = name
        self.distribution = pacal.BetaDistr(float(a),float(b))
        self.a=self.distribution.range_()[0]
        self.b=self.distribution.range_()[-1]
        self.indipendent=True
        self.sampleInit = True
        self.isScalar = False
        self.sampleSet=[]

    def execute(self):
        return self.distribution

    def getRepresentation(self):
        return "Beta ["+str(self.a)+","+str(self.b)+"]"

    def getSampleSet(self,n=100000):
        #it remembers values for future operations
        if self.sampleInit:
            self.sampleSet = self.distribution.rand(n)
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

class CustomDistr(stats.rv_continuous):
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
        #self.testpdf()
        #self.testcdf()
        #self.testicdf()
        self.distribution = MyFunDistr(CustomInterpolator(500, self.edges, self.values), breakPoints =[self.a, self.b], interpolated=False)
        self.distribution.get_piecewise_pdf()
        self.distribution=normalizeDistribution(self.distribution, init=True)

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
        return self.distribution

    def getRepresentation(self):
        return "Custom distribution in range ["+str(self.a)+","+str(self.b)+"]"

    def getSampleSet(self,n=100000):
        #it remembers values for future operations
        if self.sampleInit:
            self.sampleSet = self.rvs(size=n)
            self.sampleInit = False
        return self.sampleSet

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
        if x<=self.a:
            return 0.0
        elif x>=self.b:
            return 1.0
        else:
            index=np.digitize(x, self.edges, right=False)
            res=sum(self.area[0:index-1])+self.values[index-1]*abs(self.edges[index-1]-x)
            return res

    #def _pdf(self, x):
    #    if x<self.a or x>self.b:
    #        return 0.0
    #    index=np.digitize(x, self.edges, right=False)
    #    if index-1>=len(self.values):
    #        #can happen only when x is equal to self.b
    #        index=index-1
    #    return abs(self.values[index-1])

class U:
    def __init__(self,name,a,b):
        self.name = name
        self.distribution = pacal.UniformDistr(float(a),float(b))
        self.a=self.distribution.range_()[0]
        self.b=self.distribution.range_()[-1]
        self.indipendent = True
        self.sampleInit = True
        self.isScalar = False
        self.sampleSet=[]

    def execute(self):
        return self.distribution

    def getRepresentation(self):
        return "Uniform ["+str(self.a)+","+str(self.b)+"]"

    def getSampleSet(self,n=100000):
        #it remembers values for future operations
        if self.sampleInit:
            self.sampleSet = self.distribution.rand(n)
            self.sampleInit = False
        return self.sampleSet

class Number:
    def __init__(self, label):
        self.name = label
        self.value = float(label)
        self.distribution = pacal.ConstDistr(c = self.value)
        self.isScalar=True
        self.a=self.distribution.range_()[0]
        self.b=self.distribution.range_()[-1]

    def execute(self):
        return self.distribution

    def getRepresentation(self):
        return "Scalar("+str(self.value)+")"

    def getSampleSet(self,n=100000):
        #it remembers values for future operations
        return self.distribution.rand(n)

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