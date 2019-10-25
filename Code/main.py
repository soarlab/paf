from sympy import sign
import simple_tests
import matplotlib.pyplot
from error_model import ErrorModel
from stats import plot_error
import pacal
from fpryacc import *
import time
from regularizer import testRegularizer
from tree_model import TreeModel

matplotlib.pyplot.close("all")
filepath="./test.txt"
f= open(filepath,"r")
text=f.read()
text=text[:-1]
f.close()
myYacc=FPRyacc(text,True)
#T=TreeModel(myYacc,3,3,100)

queue=visitTree(myYacc.expression)
qDs,dDs=runAnalysis(queue,3,3,100)
plt.show()
print("\nDone\n")

def test_scalar_products():
    X=[]
    Y=[]
    for i in range(1,3):
        X.append(pacal.UniformDistr(-1,1))
        Y.append(pacal.NormalDistr())
    #error_free_scalar_product(X,Y)


def test_error_model(distribution):
    error=ErrorModel(distribution,10,-15,16,32)
    error.plot('pics/test0')


def test_plot_error(distribution):
    plot_error(distribution,10,100000)


def test_simple_tests():
    test1=simple_tests.TestUniformVariable(0,1,0.25,10)
    test1.plot_against_precision(4,32)
    test1.precision=10
    test1.plot_against_threshold()

def test_operations():
    prec=20
    emin=-15
    emax=16
    poly_prec=32
    eps=2**-prec
    X=pacal.BetaDistr(1,10)
    Y=pacal.NormalDistr(0,0.2)
    Z=pacal.BetaDistr(1,10)
    U=X*Y
    Uerr=ErrorModel(U, prec, emin, emax, poly_prec)
    print('error(U) error:  '+repr(Uerr.distribution.int_error()))
    strFile='pics/test1'
    #strFile ='pics/TH_'+repr(U.getName()).replace("'",'')+'_'+repr(prec)
    Uerr.plot(strFile)
    Ucor=U*(1+eps*Uerr.distribution)
    strFile='pics/test2'
    V=U/Z
    Verr=ErrorModel(V, prec, emin, emax, poly_prec)
    print('error(V) error:  '+repr(Verr.distribution.int_error()))
    Verr.plot(strFile)
    Vcor=V*(1+eps*Uerr.distribution)



#main:
#start = time.time()
#test_scalar_products()
#dist = pacal.UniformDistr(0 ,1 )
#test_error_model(U)
#test_plot_error(dist)
#test_operations()
#end = time.time()
#print('Elapsed time:'+repr(end - start)+'s')

#test1=simple_tests.TestUniformVariable(0,2,0.75,10)
#test1.compute()
#test2=simple_tests.TestSumUniformVariable(0,1,0,1,0.75,10)
#test2.compute()
#print(test1.error_prob)
#print(test2.error_prob)
#test1.plot_against_precision(4,32)
#test2.plot_against_precision(4,32)
