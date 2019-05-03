import simple_tests

test1=simple_tests.TestUniformVariable(0,100,32,10)

test1.plot_against_precision(4,32)
test1.precision=10
test1.plot_against_threshold()
