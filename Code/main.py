import matplotlib.pyplot
from fpryacc import *
from tree_model import TreeModel
from conditional import ConditionalError

def dependentQuantizationExecute(mantissa,exp):
    X = pacal.UniformDistr(1, 2)
    Y = pacal.UniformDistr(0, 1)
    Z = pacal.UniformDistr(1, 2)

    valX = X.rand(100000)
    valY = Y.rand(100000)
    valZ = Z.rand(100000)

    setCurrentContextPrecision(mantissa,exp)
    errors = []
    for index, val in enumerate(valX):
        x = mpfr(str(val))
        y = mpfr(str(valY[index]))
        z = mpfr(str(valZ[index]))
        resq = gmpy2.mul(gmpy2.add(x, y),z)
        res = (val + valY[index])*valZ[index]
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
mantissa=5
exp=5
text=text[:-1]
f.close()
myYacc=FPRyacc(text,True)
T = TreeModel(myYacc,mantissa,exp,100)
E = ConditionalError(T, 60, 100, (2 ** -mantissa))
plt.figure()
plt.plot(E.interpolation_points, E.get_monte_carlo_error(), linewidth=5)
dependentQuantizationExecute(mantissa,exp)
plt.show()


#>>>>>>> cb2064245eac5f5abe71f73a6679fb4e94cd80bc
#mantissa=7 #sign bit excluded
#exponent=7
#T=TreeModel(myYacc,mantissa,exponent,100)

print("\nDone\n")

