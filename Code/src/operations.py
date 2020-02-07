import math
from pychebfun import *
from regularizer import *
from project_utils import *
from gmpy2 import *


class quantizedPointMass:

   def __init__(self, wrapperInputDistribution, precision, exp):
       self.wrapperInputDistribution = wrapperInputDistribution
       self.inputdistribution = self.wrapperInputDistribution.execute()
       self.precision = precision
       self.exp = exp
       setCurrentContextPrecision(self.precision, self.exp)
       qValue = printMPFRExactly(mpfr(str(self.inputdistribution.rand(1)[0])))
       resetContextDefault()
       self.name = qValue
       self.sampleInit=True
       self.distribution = ConstDistr(float(qValue))
       self.distribution.get_piecewise_pdf()
       self.a = self.distribution.range_()[0]
       self.b = self.distribution.range_()[-1]

   def execute(self):
       return self.distribution

   def resetSampleInit(self):
       self.sampleInit=True

   def getSampleSet(self, n=100000):
       # it remembers values for future operations
       if self.sampleInit:
           # self.sampleSet = self.distribution.rand(n)
           if n <= 2:
               self.sampleSet = self.distribution.rand(n)
           else:
               self.sampleSet = self.distribution.rand(n - 2)
               self.sampleSet = np.append(self.sampleSet, [self.a, self.b])
           self.sampleInit = False
       return self.sampleSet

tmp_pdf=None
def my_tmp_pdf(t):
    return np.absolute(tmp_pdf(t))

#bins=None
#n=None
def op(t, bins, n):
    if isinstance(t, float) or isinstance(t, int) or len(t) == 1:
        if t < min(bins) or t > max(bins):
            return 0.0
        else:
            index_bin=np.digitize(t,bins)
            return abs(n[index_bin])
    else:
        res=np.zeros(len(t))
        tis=t
        for index,ti in enumerate(tis):
            if ti < min(bins) or ti > max(bins):
                res[index] = 0.0
            else:
                index_bin = np.digitize(ti, bins, right=True)
                res[index] = n[index_bin-1]
        return abs(res)
    return 0

class BinOpDist:
    """
    Wrapper class for the result of an arithmetic operation on PaCal distributions
    Warning! leftoperand and rightoperant MUST be PaCal distributions
    """
    def __init__(self, leftoperand, operator, rightoperand, poly_precision, samples_dep_op, regularize=True, convolution=True):
        self.leftoperand = leftoperand
        self.operator = operator
        self.rightoperand = rightoperand

        self.name="("+self.leftoperand.name+str(self.operator)+self.rightoperand.name+")"

        self.poly_precision=poly_precision

        self.samples_dep_op=samples_dep_op

        self.regularize = regularize
        self.convolution=convolution

        self.distribution=None
        self.distributionConv = None
        self.distributionSamp = None

        self.sampleInit=True
        self.execute()


    def executeIndependent(self):
        if self.operator == "+":
            self.distributionConv = self.leftoperand.execute() + self.rightoperand.execute()
        elif self.operator == "-":
            self.distributionConv = self.leftoperand.execute() - self.rightoperand.execute()
        elif self.operator == "*":
            self.distributionConv = self.leftoperand.execute() * self.rightoperand.execute()
        elif self.operator == "/":
            self.distributionConv = self.leftoperand.execute() / self.rightoperand.execute()
        # operator to multiply by a relative error
        elif self.operator == "*+":
            self.distributionConv = self.leftoperand.execute() * (1.0 + (self.rightoperand.eps*self.rightoperand.execute()))
        else:
            print("Operation not supported!")
            exit(-1)

        self.distributionConv.get_piecewise_pdf()

        if self.regularize:
            self.distributionConv = chebfunInterpDistr(self.distributionConv, 10)
            self.distributionConv = normalizeDistribution(self.distributionConv)

        self.aConv = self.distributionConv.range_()[0]
        self.bConv = self.distributionConv.range_()[-1]

    def operationDependent(self, elaborateBorders):
        leftOp = self.leftoperand.getSampleSet(self.samples_dep_op)
        rightOp = self.rightoperand.getSampleSet(self.samples_dep_op)

        if self.operator == "*+":
            res = np.array(leftOp) * (1 + (self.rightoperand.eps * np.array(rightOp)))
            if elaborateBorders:
                res = self.elaborateBorders(leftOp, self.operator, (1 + (self.rightoperand.eps * np.array(rightOp))), res)
        else:
            res = eval("np.array(leftOp)" + self.operator + "np.array(rightOp)")
            if elaborateBorders:
                res = self.elaborateBorders(leftOp, self.operator, rightOp, res)

        return res

    def elaborateBorders(self, leftOp, operator, rightOp, res):
        x1 = min(leftOp)
        x2 = max(leftOp)
        y1 = min(rightOp)
        y2 = max(rightOp)
        tmp_res = []
        for tmp_1 in [x1, x2]:
            for tmp_2 in [y1, y2]:
                tmp_res.append(eval(str(tmp_1)+operator+str(tmp_2)))
        res[-1]=min(tmp_res)
        res[-2]=max(tmp_res)
        return res

    def executeDependent(self):

        tmp_res = self.distributionValues

        bin_nb = int(math.ceil(math.sqrt(len(tmp_res))))

        # !!!!!!!!!!!!!!!!!!!!!!!!!!
        #Try also with bins=AUTO !!!
        #!!!!!!!!!!!!!!!!!!!!!!!!!!

        n, bins, patches = plt.hist(tmp_res, bins='auto', density=True)

        breaks=[min(bins), max(bins)]

        global tmp_pdf
        tmp_pdf = chebfun(lambda t : op(t, bins, n), domain=breaks, N=100)

        self.distributionSamp = MyFunDistr(my_tmp_pdf, breakPoints=breaks, interpolated=True)
        self.distributionSamp.get_piecewise_pdf()

        if self.regularize:
            self.distributionSamp = chebfunInterpDistr(self.distributionSamp, 10)
            self.distributionSamp = normalizeDistribution(self.distributionSamp, init=True)

        self.aSamp = self.distributionSamp.range_()[0]
        self.bSamp = self.distributionSamp.range_()[-1]

    def execute(self):
        if self.distribution==None:
            if self.convolution:
                self.executeIndependent()
                self.distributionValues = self.operationDependent(elaborateBorders=False)
                self.distribution=self.distributionConv
                self.a = self.aConv
                self.b = self.bConv
            else:
                self.distributionValues = self.operationDependent(elaborateBorders=False)
                self.executeDependent()
                self.distribution = self.distributionSamp
                self.a = self.aSamp
                self.b = self.bSamp

            self.distribution.get_piecewise_pdf()
        return self.distribution

    def getSampleSet(self,n=100000):
        #it remembers values for future operations
        if self.sampleInit:
            self.execute()
            self.sampleSet  = self.distributionValues
            self.sampleInit = False
        return self.sampleSet

    def resetSampleInit(self):
        self.sampleInit=True

class UnOpDist:
    """
    Wrapper class for the result of unary operation on a PaCal distribution
    """
    def __init__(self, operand, name, operation=None):
        if operation is None:
            self.distribution = operand.execute()
        elif operation is "exp":
            self.distribution = pacal.exp(operand.execute())
            self.distribution.get_piecewise_pdf()
        elif operation is "cos":
            self.distribution = pacal.cos(operand.execute())
            self.distribution.get_piecewise_pdf()
        elif operation is "sin":
            self.distribution = pacal.sin(operand.execute())
            self.distribution.get_piecewise_pdf()
        elif operation is "abs":
            self.distribution = abs(operand.execute())
            self.distribution.get_piecewise_pdf()
        else:
            print("Unary operation not yet supported")
            exit(-1)

        self.operand = operand
        self.name = name
        self.a = self.distribution.range_()[0]
        self.b = self.distribution.range_()[-1]

    def execute(self):
        return self.distribution

    def resetSampleInit(self):
        self.operand.sampleInit=True

    def getSampleSet(self,n=100000):
        return self.operand.getSampleSet(n)
