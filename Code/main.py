import simple_tests
from stats import plot_error
import pacal
import time

#plot_error(pacal.NormalDistr(2,0.001),11,100000)
plot_error(pacal.UniformDistr(-0.01,0.01),11,100000)



# test1=simple_tests.TestUniformVariable(0,1,0.25,10)
# test1.plot_against_precision(4,32)
# test1.precision=10
# test1.plot_against_threshold()

#test1=simple_tests.TestUniformVariable(0,2,0.75,10)
#test1.compute()
#test2=simple_tests.TestSumUniformVariable(0,1,0,1,0.75,10)
#test2.compute()
#print(test1.error_prob)
#print(test2.error_prob)
#test1.plot_against_precision(4,32)
#test2.plot_against_precision(4,32)

# start = time. time()
# #X =
# Y=0.01*UniformDistr(0,1)
# for i in range(1,80):
#     Y=Y+0.01*UniformDistr(0,1)
# Y.plot()
# #show()
# end = time. time()
# print(end - start)
