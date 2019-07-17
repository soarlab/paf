import simple_tests
import matplotlib.pyplot as plt
from error_model import ErrorModel
from stats import plot_error
import pacal
import time

def test_error_model(distribution):
    error=ErrorModel(distribution,10,-5,5)
    error.compute()


def test_plot_error(distribution):
    plot_error(distribution,10,100000)


def test_simple_tests():
    test1=simple_tests.TestUniformVariable(0,1,0.25,10)
    test1.plot_against_precision(4,32)
    test1.precision=10
    test1.plot_against_threshold()



#main:
start = time.time()
dist = pacal.UniformDistr(-50,50)
test_error_model(dist)
#test_plot_error(dist)
end = time.time()
print('Elapsed time:'+repr(end - start)+'s')

#test1=simple_tests.TestUniformVariable(0,2,0.75,10)
#test1.compute()
#test2=simple_tests.TestSumUniformVariable(0,1,0,1,0.75,10)
#test2.compute()
#print(test1.error_prob)
#print(test2.error_prob)
#test1.plot_against_precision(4,32)
#test2.plot_against_precision(4,32)
