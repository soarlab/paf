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
print("\nDone\n")