import simple_tests

# test1=simple_tests.TestUniformVariable(0,100,32,10)
# test1.plot_against_precision(4,32)
# test1.precision=10
# test1.plot_against_threshold()

test1=simple_tests.TestUniformVariable(0,2,0.25,10)
test1.compute()
test2=simple_tests.TestSumUniformVariable(0,1,0,1,0.25,10)
test2.compute()
print(test1.error_prob)
print(test2.error_prob)
