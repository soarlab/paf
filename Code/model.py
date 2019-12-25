from error_model import *
from regularizer import *
import matplotlib.pyplot as plt

class NodeManager:
    def __init__(self):
        self.id = 0
        self.nodeDict={}

    def createNode(self, value, children=None):
        if value.name in self.nodeDict:
            idtmp = self.nodeDict[value.name]
        else:
            idtmp = self.id
            self.nodeDict[value.name]=idtmp
            self.id=self.id+1

        newNode=Node(idtmp, value, children)
        if len(newNode.id) != len(set(newNode.id)):
            newNode.value.indipendent = False
            newNode.id=list(set(newNode.id))
        return newNode

class Node:
    def __init__(self, id, value, children=None):
        self.value = value
        self.father = None
        self.id = []
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
class N:
    def __init__(self,name,mu,sigma):
        self.name = name
        self.mu = mu
        self.sigma = sigma
        self.a = float("-inf")
        self.b = float("+inf")
        self.sampleInit = True
        self.sampleSet=[]
        self.indipendent=True
        self.distribution = pacal.NormalDistr(float(self.mu),float(self.sigma))

    def execute(self):
        return self.distribution

    def getRepresentation(self):
        return "Normal ["+str(self.mu)+","+str(self.sigma)+"]"

    def getSampleSet(self,n=100000):
        #it remembers values for future operations
        if self.sampleInit:
            self.sampleSet=self.distribution.rand(n)
            self.sampleInit=False
        return self.sampleSet

class B:
    def __init__(self,name,a,b):
        self.name = name
        self.distribution = pacal.BetaDistr(float(a),float(b))
        self.a=self.distribution.range_()[0]
        self.b=self.distribution.range_()[-1]
        self.indipendent=True
        self.sampleInit = True
        self.sampleSet=[]

    def execute(self):
        return self.distribution

    def getRepresentation(self):
        return "Beta ["+str(self.a)+","+str(self.b)+"]"

    def getSampleSet(self,n=100000):
        #it remembers values for future operations
        if self.sampleInit:
            self.sampleSet = self.distribution.rand(n-2)
            self.sampleSet = self.sampleSet + [self.a,self.b]
            self.sampleInit= False
        return self.sampleSet

class U:
    def __init__(self,name,a,b):
        self.name = name
        self.distribution = pacal.UniformDistr(float(a),float(b))
        self.a=self.distribution.range_()[0]
        self.b=self.distribution.range_()[-1]
        self.indipendent=True
        self.sampleInit = True
        self.sampleSet=[]

    def execute(self):
        return self.distribution

    def getRepresentation(self):
        return "Uniform ["+str(self.a)+","+str(self.b)+"]"

    def getSampleSet(self,n=100000):
        #it remembers values for future operations
        if self.sampleInit:
            self.sampleSet  = self.distribution.rand(n-2)
            self.sampleSet  = np.append(self.sampleSet, [self.a, self.b])
            self.sampleInit = False
        return self.sampleSet

class Number:
    def __init__(self, label):
        self.name = label
        self.value = float(label)
        self.distribution = pacal.ConstDistr(c = self.value)
        self.a=self.distribution.range_()[0]
        self.b=self.distribution.range_()[-1]

    def execute(self):
        return self.distribution

    def getRepresentation(self):
        return "Scalar("+str(self.value)+")"

    def getSampleSet(self,n=100000):
        #it remembers values for future operations
        return self.distribution.rand(n)

class NaiveQuantizedOperation:
    def __init__(self, name, dist, error, precision, exp):
        self.name=name
        self.dist=dist
        self.error=error
        self.precision=precision
        self.exp=exp
        self.eps = 2 ** (-self.precision)

    def execute(self):
        self.distribution=self.dependentQuantizationExecute()
        self.a = self.distribution.range_()[0]
        self.b = self.distribution.range_()[-1]
        return self.distribution

    def dependentQuantizationExecute(self):
    
        X = pacal.UniformDistr(1, 2)
        Y = pacal.UniformDistr(1, 2)
        Z = pacal.UniformDistr(1, 2)

        valX = X.rand(100000)
        valY = Y.rand(100000)
        valZ = Z.rand(100000)

        setCurrentContextPrecision(self.precision, self.exp)
        errors=[]
        for index, val in enumerate(valX):
            x = mpfr(str(val))
            y = mpfr(str(valY[index]))
            z = mpfr(str(valZ[index]))
            resq = gmpy2.mul(gmpy2.add(x,y),z)
            res = (x+y)*z
            e = (res - float(printMPFRExactly(resq))) / res #exact error of quantization
            errors.append(e)

        resetContextDefault()

        bin_nb = int(math.ceil(math.sqrt(len(res))))
        n, bins, patches = plt.hist(res, bins=bin_nb, density=1)
        plt.show()

class Operation:
    def __init__(self, leftoperand, operator, rightoperand):
        self.name = leftoperand.name + str(operator) + rightoperand.name
        self.leftoperand=leftoperand
        self.operator=operator
        self.rightoperand=rightoperand
        self.indipendent=True

    def execute(self):
        if self.indipendent:
            indipendentExecute()
            dependentExecute()
            return self.distribution
        else:
            self.dependentExecute()
            return self.approx_distribution

def executeOperation(leftOperand, operatorString, rightOperand):
    if operatorString == "+":
        distribution = leftOperand.distribution + rightOperand.distribution
    elif operatorString == "-":
        distribution = leftOperand.distribution - rightOperand.distribution
    elif operatorString == "*":
        distribution = leftOperand.distribution * rightOperand.distribution
    elif operatorString == "/":
        distribution = leftOperand.distribution / rightOperand.distribution
    else:
        print("Operation not supported!")
        exit(-1)
    return distribution

def dependentExecute(leftoperand, operator, rightoperand):
    leftOp = leftoperand.getSampleSet()
    rightOp = rightoperand.getSampleSet()
    res = eval("np.array(leftOp)"+operator+"np.array(rightOp)")
    bin_nb = int(math.ceil(math.sqrt(len(res))))
    n, bins, patches = plt.hist(res, bins=bin_nb, density=1)

    bins, n = (list(t) for t in zip(*sorted(zip(bins, n))))
    enumBins=enumerate(bins)
    pdf = chebfun(lambda t: op(t,bins,enumBins), domain= [min(bins), max(bins)], N=100)
    approx_distribution = FunDistr(pdf)
    approx_distribution.init_piecewise_pdf()
    return approx_distribution

def op(t,bins,n):
    if t<min(bins) or t>min(bins):
        return 0
    else:
        idx = (np.abs(bins - t)).argmin()
        return n[idx]

def indipendentExecute(leftoperand, operator, rightoperand):
    distribution = executeOperation(leftoperand, operator, rightoperand)
    return distribution