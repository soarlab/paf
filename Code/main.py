import matplotlib.pyplot
from fpryacc import *
from tree_model import TreeModel
from conditional import ConditionalError

def dependentQuantizationExecute():
    X = pacal.UniformDistr(1, 2)
    Y = pacal.UniformDistr(1, 2)
    Z = pacal.UniformDistr(1, 2)

    valX = X.rand(1000000)
    valY = Y.rand(1000000)
    valZ = Z.rand(1000000)

    setCurrentContextPrecision(3,3)
    errors = []
    for index, val in enumerate(valX):
        x = mpfr(str(val))
        y = mpfr(str(valY[index]))
        z = mpfr(str(valZ[index]))
        resq = gmpy2.mul(gmpy2.add(x, y), z)
        res = (val + valY[index]) * valZ[index]
        e = (res - float(printMPFRExactly(resq))) / res  # exact error of quantization
        errors.append(e)

    resetContextDefault()

    bin_nb = int(math.ceil(math.sqrt(len(errors))))
    n, bins, patches = plt.hist(errors, bins=bin_nb, density=1)
    plt.show()

matplotlib.pyplot.close("all")
filepath="./test.txt"
f= open(filepath,"r")
text=f.read()
text=text[:-1]
f.close()
myYacc=FPRyacc(text,True)
T = TreeModel(myYacc,3,3,60)
E = ConditionalError(T, 60, 100, 3)
plt.figure()
plt.plot(E.interpolation_points, E.get_monte_carlo_error())
dependentQuantizationExecute()
plt.show()




#>>>>>>> cb2064245eac5f5abe71f73a6679fb4e94cd80bc
#mantissa=7 #sign bit excluded
#exponent=7
#T=TreeModel(myYacc,mantissa,exponent,100)

print("\nDone\n")

