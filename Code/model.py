from sympy import sign

import simple_tests
import matplotlib.pyplot as plt
from error_model import ErrorModel
from stats import plot_error
import pacal
import time


def visitTree(node):
    queue=[node]
    i=0
    while i<len(queue):
        if node.leaf == True:
            pass
        else:
            queue.append(node.children)
        i = i + 1
    queue=list(reversed(queue))
    return queue

def runAnalysis(queue,prec,exp,poly_prec):
    emin=-(2**exp)+1
    emax=2**exp
    eps=2**-prec
    quantizedDistributions = {}
    doubleDistributions = {}
    for elem in queue:
        name= elem.value.name
        if not name in quantizedDistributions:
            doubleDistribution = elem.value.execute()
            doubleDistributions[name] = doubleDistribution
            Uerr = ErrorModel(doubleDistribution, prec, emin, emax, poly_prec)
            quantizedDistributions[name] = doubleDistribution * (1 + eps * Uerr.distribution)

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
        self.mu=mu
        self.sigma=sigma
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

class Operation:
    def __init__(self, leftNode, operator, rightNode):
        self.name = leftNode.value.name+str(operator)+rightNode.value.name
        self.leftnode=leftNode
        self.operator=operator
        self.rightnode=rightNode

        if operator=="+":
            self.distribution=leftNode.value.distribution+rightNode.value.distribution
        elif operator=="-":
            self.distribution=leftNode.value.distribution-rightNode.value.distribution
        elif operator=="*":
            self.distribution=leftNode.value.distribution*rightNode.value.distribution
        elif operator=="/":
            self.distribution=leftNode.value.distribution/rightNode.value.distribution

    def execute(self):
        return self.distribution