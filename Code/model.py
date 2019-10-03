from sympy import sign

import simple_tests
import matplotlib.pyplot as plt
from error_model import *
from stats import plot_error
import pacal
params.general.parallel=True

from pacal import *
import time
from pychebfun import *

import matplotlib.pyplot as plt
import numpy as np

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

def runAnalysis(queue,prec,exp,poly_prec):
    eps=2**(-prec)
    quantizedDistributions = {}
    quantizedDistributionsNaive={}
    doubleDistributions = {}
    plt.close('all')
    for elem in queue:
        name= elem.value.name
        if not name in quantizedDistributions:
            if isinstance(elem.value, Operation):
                doubleDistribution = elem.value.execute()
                doubleDistributions[name] = doubleDistribution
                plt.figure("DoublePrecision: "+name)
                doubleDistributions[name].plot()
                plt.figure("Relative Error Distribution: " + name)
                nameD, leftoperandD, operator, rightoperandD = elem.value.extractInfoForQuantization()
                QleftDistribution = quantizedDistributions[leftoperandD.name]
                QrightDistribution = quantizedDistributions[rightoperandD.name]
                quantizedOperation = QuantizedOperation(nameD,QleftDistribution, operator, QrightDistribution)
                quantizedDistribution = quantizedOperation.execute()
                Uerr = ErrorModel(quantizedOperation, prec, exp, poly_prec)
                Uerr.distribution.plot()
                # plt.figure("Relative Error Naive: " + name)
                errModelNaive = ErrorModelNaive(quantizedDistribution, prec, 100000)
                x_values, error_values = errModelNaive.compute_naive_error()
                errModelNaive.plot_error(error_values, "Relative Error Distribution: " + name)
                plt.figure("Quantized Distribution: " + name)
                quantizedDistributions[name] = quantizedDistribution * (1 + (eps * Uerr.distribution))
                (quantizedDistributions[name]).plot()
                #plt.pause(0.05)
                plt.show()
            else:
                doubleDistribution = elem.value.execute()
                doubleDistributions[name] = doubleDistribution
                #plt.figure("DoublePrecision: "+name)
                #plt.title("DoublePrecision: "+name)
                #doubleDistributions[name].plot()
                #plt.figure("Relative Err. Distr: "+name)
                #plt.title("Relative Err. Distr.: "+name+", Cheb_Poly_Prec: "+str(poly_prec))
                Uerr = ErrorModel(elem.value, prec, exp, poly_prec)
                #Uerr.distribution.plot()
                #plt.figure("Relative Error Naive: " + name)
                errModelNaive = ErrorModelNaive(doubleDistributions[name], prec, 100000)
                x_values, error_values =errModelNaive.compute_naive_error()
                #errModelNaive.plot_error(error_values,"Relative Err. Distr: "+name)
                #plt.figure("Quantized Distribution: "+name)
                quantizedDistributions[name] = doubleDistribution*(1 + (eps * Uerr.distribution))
                quantizedDistributions[name].init_piecewise_pdf()
                #(quantizedDistributions[name]).plot()
                #plt.scatter(x=[float(quantizedDistributions[name].a), float(quantizedDistributions[name].b)], y=[0, 0], c='r', marker="|", s=1000)
                #plt.annotate(str(float(quantizedDistributions[name].a)),(float(quantizedDistributions[name].a)-0.05,0.05))
                #plt.annotate(str(float(quantizedDistributions[name].b)),(float(quantizedDistributions[name].b),0.05))
                #plt.title("Quantized Distribution: "+name+"; Mantissa: "+str(prec)+"bit, Exp: "+str(exp)+"bit, Segments: "+str(len(quantizedDistributions[name].get_piecewise_pdf().segments)))
                #plt.xticks(list(plt.xticks()[0]) + [float(quantizedDistributions[name].a), float(quantizedDistributions[name].b)])
                #plt.pause(0.05)
                #plt.show()

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

class B:
    def __init__(self,name,a,b):
        self.name = name
        self.a=a
        self.b=b
        self.distribution = pacal.BetaDistr(float(self.a),float(self.b))

    def execute(self):
        return self.distribution

class U:
    def __init__(self,name,a,b):
        self.name = name
        self.a=a
        self.b=b
        self.distribution = pacal.UniformDistr(float(self.a),float(self.b))

    def execute(self):
        return self.distribution

class Number:
    def __init__(self, label):
        self.name = label
        self.value = float(label)

    def execute(self):
        pass

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
        self.a = self.distribution.a
        self.b = self.distribution.b

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
        self.a = self.distribution.a
        self.b = self.distribution.b

    def execute(self):
        return self.distribution

    def extractInfoForQuantization(self):
        return self.name, self.leftoperand, self.operator, self.rightoperand
