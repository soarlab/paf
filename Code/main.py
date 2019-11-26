import matplotlib.pyplot
from fpryacc import *
from tree_model import *

matplotlib.pyplot.close("all")
filepath="./test.txt"
f= open(filepath,"r")
text=f.read()
text=text[:-1]
f.close()
myYacc=FPRyacc(text,True)

mantissa=7 #sign bit excluded
exponent=7
T=TreeModel(myYacc,mantissa,exponent,100)

print("\nDone\n")

