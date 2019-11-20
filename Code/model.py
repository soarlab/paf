from sympy import sign

import simple_tests
import matplotlib.pyplot as plt
from error_model import *
from stats import plot_error
import pacal
#params.general.parallel=True
import regularizer
from regularizer import *
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import utils
from utils import *

def visitTree(node):
    queue=[node]
    i=0
    while i<len(queue):
        tmpNode=queue[i]
        if tmpNode.leaf == True:
            pass
        else:
            queue=queue+tmpNode.children
        i = i + 1
    queue=list(reversed(queue))
    return queue

def plotDistribution(name,distribution,title):
    plt.figure(name)
    plt.title(title, fontsize=15)
    distribution.plot()

def plotTicks(figureName,distribution):
    plt.figure(figureName)
    minVal = distribution.range_()[0]
    maxVal = distribution.range_()[1]
    plt.scatter(x=[minVal, maxVal], y=[0, 0], c='r', marker="|", s=1000)
    plt.annotate(str(minVal),(minVal-0.05,0.05))
    plt.annotate(str(maxVal),(maxVal,0.05))
    plt.xlabel('Distribution Range', fontsize=15)
    plt.ylabel('PDF', fontsize=15)
    #plt.xticks(list(plt.xticks()[0]) + [minVal, maxVal])

def plotBoundsWhenOutOfRange(figureName,distribution, mantissa,exponent):
    minVal = min(distribution.get_piecewise_pdf().breaks)
    maxVal = max(distribution.get_piecewise_pdf().breaks)
    res=checkBoundsOutOfRange(minVal, maxVal,mantissa,exponent)
    if (res[0]!=0):
        plt.figure(figureName)
        plt.scatter(x=float(res[0]), y=0, c='b', marker="|", s=1000)
        plt.annotate(str(float(res[0])), (float(res[0]) - 0.05, 0.05))
        plt.xticks(list(plt.xticks()[0]) + [float(res[0])])
        print("Probability of overflow (negative) for " + figureName + ": " + str(distribution.get_piecewise_cdf()(res[0])))

    if (res[1]!=0):
        plt.figure(figureName)
        plt.scatter(x=float(res[1]), y=0, c='b', marker="|", s=1000)
        plt.annotate(str(float(res[1])), (float(res[1]) - 0.05, 0.05))
        plt.xticks(list(plt.xticks()[0]) + [float(res[1])])
        print("Probability of overflow (positive) for "+figureName+": "+str(1-distribution.get_piecewise_cdf()(res[1])))

def runAnalysis(queue,prec,exp,poly_prec):
    eps=2**(-prec)
    quantizedDistributions = {}
    doubleDistributions = {}

    for elem in queue:
        name= elem.value.name
        if not name in quantizedDistributions:
            if isinstance(elem.value, Operation):
                doubleDistribution = elem.value.execute()
                doubleDistributions[name] = doubleDistribution
                DoubleOp="Double "+elem.value.name+" = ["+str(elem.value.a)+", "+str(elem.value.b)+"]"
                plotDistribution("DoublePrecision: "+name, doubleDistribution, DoubleOp)
                nameD, leftoperandD, operator, rightoperandD = elem.value.extractInfoForQuantization()
                QleftDistribution = quantizedDistributions[leftoperandD.name]
                QrightDistribution = quantizedDistributions[rightoperandD.name]
                quantizedOperation = QuantizedOperation(nameD,QleftDistribution, operator, QrightDistribution)
                quantizedDistribution = quantizedOperation.execute()
                Uerr = ErrorModel(quantizedOperation, prec, exp, poly_prec)
                plotDistribution("Relative Error Distribution: " + name, Uerr.distribution, "Err.Distr. [-1, 1]")
                errModelNaive = ErrorModelNaive(quantizedDistribution, prec, 100000)
                x_values, error_values = errModelNaive.compute_naive_error()
                errModelNaive.plot_error(error_values, "Relative Error Distribution: " + name)
                quantizedDistributions[name] = quantizedDistribution * (1 + (eps * Uerr.distribution))
                QuantizedOp="Quantized "+elem.value.name+" = ["+str(quantizedDistributions[name].range_()[0])+", "+str(quantizedDistributions[name].range_()[1])+"]"
                plotDistribution("Quantized Distribution: " + name, quantizedDistributions[name], QuantizedOp)
                plotTicks("Quantized Distribution: " + name, quantizedDistributions[name])
                plotBoundsWhenOutOfRange("Quantized Distribution: " + name, quantizedDistributions[name], prec, exp)
                quantizedDistributions[name]=chebfunInterpDistr(quantizedDistributions[name], 10)
                #quantizedDistributions[name]=normalizeDistribution(quantizedDistributions[name])
            else:
                doubleDistribution = elem.value.execute()
                doubleDistributions[name] = doubleDistribution
                plotDistribution("DoublePrecision: "+name, doubleDistribution, elem.value.getRepresentation())
                Uerr = ErrorModel(elem.value, prec, exp, poly_prec)
                plotDistribution("Relative Error Distribution: " + name, Uerr.distribution, "Err.Distr. [-1, 1]")
                errModelNaive = ErrorModelNaive(doubleDistributions[name], prec, 100000)
                x_values, error_values =errModelNaive.compute_naive_error()
                errModelNaive.plot_error(error_values,"Relative Error Distribution: " + name)
                quantizedDistributions[name] = doubleDistribution*(1.0 + (eps * Uerr.distribution))
                QuantizedOp="Quantized "+elem.value.name+" = ["+str(quantizedDistributions[name].range_()[0])+", "+str(quantizedDistributions[name].range_()[1])+"]"
                plotDistribution("Quantized Distribution: " + name, quantizedDistributions[name], QuantizedOp)
                plotTicks("Quantized Distribution: " + name, quantizedDistributions[name])
                plotBoundsWhenOutOfRange("Quantized Distribution: " + name, quantizedDistributions[name], prec, exp)
                quantizedDistributions[name]=chebfunInterpDistr(quantizedDistributions[name], 10)
                #quantizedDistributions[name]=normalizeDistribution(quantizedDistributions[name])

    return doubleDistributions,quantizedDistributions

class Node:
    def __init__(self, value, children=None):
        self.value = value
        if children:
            self.leaf=False
            self.children = children
        else:
            self.leaf = True
            self.children = []

class N:
    def __init__(self,name,mu,sigma):
        self.name = name
        self.mu = mu
        self.sigma = sigma
        self.a = float("-inf")
        self.b = float("+inf")
        self.distribution = pacal.NormalDistr(float(self.mu),float(self.sigma))

    def execute(self):
        return self.distribution

    def getRepresentation(self):
        return "Normal ["+str(self.mu)+","+str(self.sigma)+"]"

class B:
    def __init__(self,name,a,b):
        self.name = name
        self.distribution = pacal.BetaDistr(float(a),float(b))
        self.a=self.distribution.range()[0]
        self.b=self.distribution.range()[-1]

    def execute(self):
        return self.distribution

    def getRepresentation(self):
        return "Beta ["+str(self.a)+","+str(self.b)+"]"

class U:
    def __init__(self,name,a,b):
        self.name = name
        self.distribution = pacal.UniformDistr(float(a),float(b))
        self.a=self.distribution.range()[0]
        self.b=self.distribution.range()[-1]

    def execute(self):
        return self.distribution

    def getRepresentation(self):
        return "Uniform ["+str(self.a)+","+str(self.b)+"]"

class Number:
    def __init__(self, label):
        self.name = label
        self.value = float(label)

    def execute(self):
        pass

    def getRepresentation(self):
        return "Scalar("+str(self.value)+")"

class QuantizedOperation:
    def __init__(self, name, leftDistribution, operator, rightDistribution):
        self.name=name
        self.leftDistribution=leftDistribution
        self.operator=operator
        self.rightDistribution=rightDistribution

        if operator=="+":
            self.distribution=self.leftDistribution+self.rightDistribution
        elif operator=="-":
            self.distribution=self.leftDistribution-self.rightDistribution
        elif operator=="*":
            self.distribution=self.leftDistribution*self.rightDistribution
        elif operator=="/":
            self.distribution=self.leftDistribution/self.rightDistribution
        else:
            print ("Operation not supported!")
            exit(-1)
        ####################################
        self.a = self.distribution.range()[0]
        self.b = self.distribution.range()[-1]

    def execute(self):
        return self.distribution

class Operation:
    def __init__(self, leftoperand, operator, rightoperand, parenthesis):
        TMPname = leftoperand.name + str(operator) + rightoperand.name
        if parenthesis:
            self.name="("+TMPname+")"
        else:
            self.name=TMPname

        self.leftoperand=leftoperand
        self.operator=operator
        self.rightoperand=rightoperand

        if operator=="+":
            self.distribution=self.leftoperand.distribution+self.rightoperand.distribution
        elif operator=="-":
            self.distribution=self.leftoperand.distribution-self.rightoperand.distribution
        elif operator=="*":
            self.distribution=self.leftoperand.distribution*self.rightoperand.distribution
        elif operator=="/":
            self.distribution=self.leftoperand.distribution/self.rightoperand.distribution
        else:
            print ("Operation not supported!")
            exit(-1)
        ####################################
        self.a = self.distribution.range()[0]
        self.b = self.distribution.range()[-1]

    def execute(self):
        return self.distribution

    def extractInfoForQuantization(self):
        return self.name, self.leftoperand, self.operator, self.rightoperand
