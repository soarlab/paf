import simple_tests
import matplotlib.pyplot as plt
from error_model import ErrorModel
from stats import plot_error
import pacal
import time

def test_error_model():
    error=ErrorModel(pacal.UniformDistr(-100,100),10,-14,15)



def test_plot_error():
    plot_error(pacal.NormalDistr(0.9,0.1),11,100000)
    plot_error(pacal.UniformDistr(0.9,1),11,100000)

def test_simple_tests():
    test1=simple_tests.TestUniformVariable(0,1,0.25,10)
    test1.plot_against_precision(4,32)
    test1.precision=10
    test1.plot_against_threshold()



#main:
start = time.time()
test_error_model()
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
