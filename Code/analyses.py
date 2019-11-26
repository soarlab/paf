from error_model import *
from pacal import *
params.general.parallel=True
from regularizer import *
import matplotlib.pyplot as plt
from utils import *

plt.rcParams.update({'font.size': 40})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'legend.frameon': False})
plt.rcParams.update({'legend.handletextpad': 0.1})
plt.rcParams.update({'legend.labelspacing': 0.5})
plt.rcParams.update({'axes.labelpad': 20})
plt.rcParams.update({'legend.loc':'upper right'})
#plt.rcParams.update({'bbox_to_anchor':(0.0,0.0)})


def visitTree(node):
    queue=[node]
    i=0
    while i<len(queue):
        tmpNode=queue[i]
        if tmpNode.leaf == True:
            print (tmpNode.value.name)
            pass
        else:
            queue=queue+tmpNode.children
        i = i + 1
    queue=list(reversed(queue))
    return queue

def plotDistribution(name,distribution,title,label=None):
    plt.figure(name)
    plt.title(title)
    plt.xlabel('Distribution Range')
    plt.ylabel('PDF')
    plt.legend(bbox_to_anchor=(-0.5, -0.5))
    #plt.legend(bbox_to_anchor=(2, 2))#, 0.5, 0.5))#frameon=False, handletextpad=0.1, labelspacing=1.0)
    distribution.plot(linewidth=1, label=label)

def plotTicks(figureName,distribution,ticks=None,label=""):
    if not ticks==None:
        minVal = ticks[0]
        maxVal = ticks[1]
        plt.figure(figureName)
        plt.scatter(x=[minVal, maxVal], y=[0, 0], c='b', marker="|", label=label, linewidth=8, s=1000)
        plt.legend()#frameon=False, handletextpad=0.1, labelspacing=1.0)#, loc='upper right')
        plt.xlabel('Distribution Range')
        plt.ylabel('PDF')
    else:
        minVal = distribution.range_()[0]
        maxVal = distribution.range_()[1]
        labelMinVal = str("%.2f" % distribution.range_()[0])
        labelMaxVal = str("%.2f" % distribution.range_()[1])
        plt.figure(figureName)
        plt.scatter(x=[minVal, maxVal], y=[0, 0], c='r', marker="|", label="PM: ["+labelMinVal+","+labelMaxVal+"]", linewidth=8, s=1000)
        plt.legend()#frameon=False, handletextpad=0.1, labelspacing=1.0)#, loc='upper right')
        plt.xlabel('Distribution Range')
        plt.ylabel('PDF')

def plotBoundsWhenOutOfRange(figureName,distribution, res):
    if (res[0]!=0):
        label=str("%.5f" % distribution.get_piecewise_pdf().integrate(float("-inf"), res[0]))
        plt.figure(figureName)
        plt.scatter(x=float(res[0]), y=0, c='brown', marker="|", label="Prob:" + label, linewidth=8, s=1000)
        plt.legend()#frameon=False, handletextpad=0.1, labelspacing=1.0)#, loc='upper right')
        print("Probability of overflow (negative) for " + figureName + ": " + str(distribution.get_piecewise_cdf()(res[0])))

    if (res[1]!=0):
        label=str("%.5f" % (1.0 - distribution.get_piecewise_pdf().integrate(float("-inf"), res[1])))
        plt.figure(figureName)
        plt.scatter(x=float(res[1]), y=0, c='green', marker="|", label="Prob: " + label, linewidth=8, s=1000)
        plt.legend()#frameon=False, handletextpad=0.1, labelspacing=2.0)#, loc='upper right')
        print("Probability of overflow (positive) for "+figureName+": "+str(1-distribution.get_piecewise_cdf()(res[1])))

def runAnalysisRange(queue,prec,exp,poly_prec):
    eps=2**(-prec)
    quantizedDistributions = {}
    doubleDistributions = {}
    errorDistributions = {}
    format="Custom (m: "+str(prec)+"bits, e: "+str(exp)+"bits)"
    for elem in queue:
        name= elem.value.name
        if not name in quantizedDistributions:
            if isinstance(elem.value, Operation):
                doubleDistribution = elem.value.execute()
                doubleDistributions[name] = doubleDistribution
                DoubleOp="Double (m:52bits, e:11bits) \n"+elem.value.name
                plotDistribution("DoublePrecision: "+name, doubleDistribution, DoubleOp)
                plotTicks("DoublePrecision: "+name, doubleDistribution)
                nameD, leftoperandD, operator, rightoperandD = elem.value.extractInfoForQuantization()
                QleftDistribution = quantizedDistributions[leftoperandD.name]
                QrightDistribution = quantizedDistributions[rightoperandD.name]
                quantizedOperation = QuantizedOperation(nameD,QleftDistribution, operator, QrightDistribution)
                quantizedDistribution = quantizedOperation.execute()
                Uerr = ErrorModel(quantizedOperation, prec, exp, poly_prec)
                plotDistribution("Relative Error Distribution: " + name, Uerr.distribution, format+", Err.Distr. [-1eps, 1eps]\n"+name, label="Prob. Model")
                errModelNaive = ErrorModelNaive(quantizedDistribution, prec, 100000)
                x_values, error_values = errModelNaive.compute_naive_error()
                errModelNaive.plot_error(error_values, "Relative Error Distribution: " + name)
                errorDistributions[name] = Uerr.distribution
                quantizedDistributions[name] = quantizedDistribution * (1 + (eps * Uerr.distribution))
                QuantizedOp=format+"\n"+elem.value.name
                labelProbModel="["+str(quantizedDistributions[name].range_()[0])+", "+str(quantizedDistributions[name].range_()[1])+"]"
                plotDistribution("Quantized Distribution: " + name, quantizedDistributions[name], QuantizedOp)
                plotTicks("Quantized Distribution: " + name, quantizedDistributions[name], label=labelProbModel)
                plotTicks("Quantized Distribution: " + name, None, ticks=[None,None], label="FPT: [-inf,+inf]")
                quantizedDistributions[name]=chebfunInterpDistr(quantizedDistributions[name], 10)
                quantizedDistributions[name]=normalizeDistribution(quantizedDistributions[name])
                res=getBoundsWhenOutOfRange(quantizedDistributions[name], prec, exp)
                plotBoundsWhenOutOfRange("Quantized Distribution: " + name, quantizedDistributions[name],res)

            else:
                doubleDistribution = elem.value.execute()
                doubleDistributions[name] = doubleDistribution
                plotDistribution("DoublePrecision: "+name, doubleDistribution, elem.value.getRepresentation())
                Uerr = ErrorModel(elem.value, prec, exp, poly_prec)
                plotDistribution("Relative Error Distribution: " + name, Uerr.distribution, format+name+": Err.Distr. [-1, +1](eps)", label="Prob. Model")
                errModelNaive = ErrorModelNaive(doubleDistributions[name], prec, 100000)
                x_values, error_values = errModelNaive.compute_naive_error()
                errModelNaive.plot_error(error_values, "Relative Error Distribution: " + name)
                quantizedDistributions[name] = doubleDistribution*(1.0 + (eps*Uerr.distribution))
                QuantizedOp=format+"\n"+elem.value.name+" = ["+str(quantizedDistributions[name].range_()[0])+", "+str(quantizedDistributions[name].range_()[1])+"]"
                plotDistribution("Quantized Distribution: " + name, quantizedDistributions[name], QuantizedOp)
                plotTicks("Quantized Distribution: " + name, quantizedDistributions[name])
                quantizedDistributions[name]=chebfunInterpDistr(quantizedDistributions[name], 10)
                quantizedDistributions[name]=normalizeDistribution(quantizedDistributions[name])
                res=getBoundsWhenOutOfRange(quantizedDistributions[name], prec, exp)
                plotBoundsWhenOutOfRange("Quantized Distribution: " + name, quantizedDistributions[name],res)

    return doubleDistributions,quantizedDistributions

def runAnalysisFinal(queue,prec,exp,poly_prec):
    eps=2**(-prec)
    quantizedDistributions = {}
    doubleDistributions = {}
    errorDistributions = {}

    for elem in queue:
        name= elem.value.name
        if not name in quantizedDistributions:
            if isinstance(elem.value, Operation):
                doubleDistribution = elem.value.execute()
                doubleDistributions[name] = doubleDistribution
                DoubleOp="Double "+elem.value.name+" = ["+str(elem.value.a)+", "+str(elem.value.b)+"]"
                plotDistribution("DoublePrecision: "+name, doubleDistribution, DoubleOp)
                nameD, leftoperandD, operator, rightoperandD = elem.value.extractInfoForQuantization()

                error= 2*errorDistributions[leftoperandD.name]+errorDistributions[leftoperandD.name]*errorDistributions[leftoperandD.name]
                error.init_piecewise_pdf()
                plotDistribution("Relative Error Distribution: " + name, error, "Err.Distr.")
                print("Done")

            else:
                doubleDistribution = elem.value.execute()
                doubleDistributions[name] = doubleDistribution
                plotDistribution("DoublePrecision: "+name, doubleDistribution, elem.value.getRepresentation())
                Uerr = ErrorModel(elem.value, prec, exp, poly_prec)
                errorDistributions[name]=eps*Uerr.distribution
                error=eps*Uerr.distribution
                plotDistribution("Relative Error Distribution: " + name, error, "Err.Distr.")
                print ("Done")
                #errModelFinal=(eps * Uerr.distribution) + (eps * Uerr.distribution) + (eps * Uerr.distribution)*(eps * Uerr.distribution)
                #plotDistribution("Relative Error Distribution: " + name, errModelFinal, "Err.Distr.")
                #errModelNaive = ErrorModelNaive(doubleDistributions[name], prec, 100000)
                #x_values, error_values = errModelNaive.compute_naive_error()
                #errModelNaive.plot_error(error_values, "Relative Error Distribution: " + name)
                #quantizedDistributions[name] = doubleDistribution*(1.0 + errModelFinal)
                #QuantizedOp="Quantized "+elem.value.name+" = ["+str(quantizedDistributions[name].range_()[0])+", "+str(quantizedDistributions[name].range_()[1])+"]"
                #plotDistribution("Quantized Distribution: " + name, quantizedDistributions[name], QuantizedOp)
                #plotTicks("Quantized Distribution: " + name, quantizedDistributions[name])
                #plotBoundsWhenOutOfRange("Quantized Distribution: " + name, quantizedDistributions[name], prec, exp)
                #quantizedDistributions[name]=chebfunInterpDistr(quantizedDistributions[name], 10)
                #quantizedDistributions[name]=normalizeDistribution(quantizedDistributions[name])

    return doubleDistributions,quantizedDistributions