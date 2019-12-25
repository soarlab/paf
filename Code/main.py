import matplotlib.pyplot
from fpryacc import *
from tree_model import TreeModel

matplotlib.pyplot.close("all")
filepath="./test.txt"
f= open(filepath,"r")
text=f.read()
#mantissa with implicit bit of sign
#gmpy2 set precision=p includes also sign bit.
mantissa=11
exp=5
print(computeLargestPositiveNumber(mantissa, exp))
text=text[:-1]
f.close()
myYacc=FPRyacc(text, True)
T = TreeModel(myYacc, mantissa, exp, 100)
T.plot_range_analysis(100000,T.tree.root_value[0].name)
#T.plot_empirical_error_distribution(100000,"error_dist")
# E = ConditionalError(T, 60, 100, (2 ** -mantissa))
# plt.figure()
# plt.plot(E.interpolation_points, E.get_monte_carlo_error(), linewidth=5)
# dependentQuantizationExecute(mantissa,exp)
# plt.show()
#>>>>>>> cb2064245eac5f5abe71f73a6679fb4e94cd80bc
#mantissa=7 #sign bit excluded
#exponent=7
#T=TreeModel(myYacc,mantissa,exponent,100)

print("\nDone\n")