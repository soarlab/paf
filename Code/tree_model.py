from model import *
from error_model import *
from regularizer import *
import time

plt.rcParams.update({'font.size': 30})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'legend.frameon': False})
plt.rcParams.update({'legend.handletextpad': 0.1})
plt.rcParams.update({'legend.labelspacing': 0.5})
plt.rcParams.update({'axes.labelpad': 20})
plt.rcParams.update({'legend.loc':'best'})

def copy_tree(my_tree):
    if my_tree.leaf:
        copied_tree = BinaryTree(my_tree.value.name, my_tree.value)
    else:
        if my_tree.value.operator in ["exp", "cos", "sin"]:
            copied_tree = UnaryTree(my_tree.value.operator, None, copy_tree(my_tree.children[0]))
        else:
            copied_tree = BinaryTree(my_tree.value.operator, None,
                                     copy_tree(my_tree.children[0]),
                                     copy_tree(my_tree.children[1]), my_tree.value.indipendent)
    return copied_tree


class BinaryTree(object):

    def __init__(self, name, value, left=None, right=None, convolution=True):
        self.root_name = name
        self.root_value = value
        self.left = left
        self.right = right
        self.convolution = convolution

class UnaryTree(object):

    def __init__(self, name, value, onlychild):
        self.root_name = name
        self.root_value = value
        self.left = onlychild
        self.right = None

def isPointMassDistr(dist):
    if dist.distribution.range_()[0] == dist.distribution.range_()[-1]:
        return True
    return False

class Triple:
    def __init__(self, dist, error, qdist):
        self.dist=dist
        self.error=error
        self.qdist=qdist


class DistributionsManager:
    def __init__(self):
        self.errordictionary = {}
        self.distrdictionary = {}

    def createErrorModel(self, wrapDist, precision, exp, pol_prec, typical=True):
        if typical:
            if wrapDist.name in self.errordictionary:
                return self.errordictionary[wrapDist.name]
            else:
                tmp=TypicalErrorModel(precision, exp, pol_prec)
                self.errordictionary[wrapDist.name] = tmp
                return tmp
        else:
            if wrapDist.name in self.errordictionary:
                return self.errordictionary[wrapDist.name]
            else:
                tmp=ErrorModel(wrapDist, precision, exp, pol_prec)
                self.errordictionary[wrapDist.name]=tmp
                return tmp

    def createBinOperation(self, leftoperand, operator, rightoperand, poly_precision, regularize=True, convolution=True):
        name="("+leftoperand.name+str(operator)+rightoperand.name+")"
        if name in self.distrdictionary:
            return self.distrdictionary[name]
        else:
            tmp=BinOpDist(leftoperand, operator, rightoperand, poly_precision, regularize, convolution)
            self.distrdictionary[name]=tmp
            return tmp

    def createUnaryOperation(self, operand, name, operation=None):
        if operation is not None:
            tmp_name=name+"("+operand.name+")"
        else:
            tmp_name=name
        if tmp_name in self.distrdictionary:
            return  self.distrdictionary[tmp_name]
        else:
            tmp=UnOpDist(operand, tmp_name, operation)
            self.distrdictionary[tmp_name]=tmp
            return tmp

class TreeModel:

    def __init__(self, my_yacc, precision, exp, poly_precision, initialize=True):
        self.initialize = initialize
        self.precision = precision
        self.exp = exp
        self.poly_precision = poly_precision
        # Copy structure of the tree from my_yacc
        self.tree = copy_tree(my_yacc.expression)
        # Evaluate tree
        self.eps = 2 ** (-self.precision)
        self.manager=DistributionsManager()
        self.evaluate(self.tree)

    def evaluate(self, tree):
        """ Recursively populate the Tree with the triples
        (distribution, error distribution, quantized distribution) """
        # Test if we're at a leaf
        if tree.root_value is not None:
            # Non-quantized distribution
            dist = self.manager.createUnaryOperation(tree.root_value, tree.root_name)
            # initialize=True means we quantize the inputs
            if self.initialize:
                # Compute error model
                if isPointMassDistr(dist):
                    error = ErrorModelPointMass(dist, self.precision, self.exp)
                    quantized_distribution = quantizedPointMass(dist,self.precision, self.exp)
                else:
                    error = self.manager.createErrorModel(dist, self.precision, self.exp, self.poly_precision)
                    quantized_distribution = self.manager.createBinOperation(dist, "*+", error, self.poly_precision)
            # Else we leave the leaf distribution unchanged
            else:
                error = 0
                quantized_distribution = dist

        # If not at a leaf we need to get the distribution and quantized distributions of the children nodes.
        # Then, check the operation. For each operation the template is the same:
        # dist will be the non-quantized operation the non-quantized children nodes
        # qdist will be the non-quantized operation on the quantized children nodes
        # quantized_distribution will be the quantized operation on the quantized children nodes

        elif tree.left is not None and tree.right is not None:

            self.evaluate(tree.left)
            self.evaluate(tree.right)

            dist  = self.manager.createBinOperation(tree.left.root_value[0], tree.root_name, tree.right.root_value[0], self.poly_precision, convolution=tree.convolution)
            qdist = self.manager.createBinOperation(tree.left.root_value[2], tree.root_name, tree.right.root_value[2], self.poly_precision, convolution=tree.convolution)

            if isPointMassDistr(dist):
                error = ErrorModelPointMass(qdist, self.precision, self.exp)
                quantized_distribution = quantizedPointMass(dist, self.precision, self.exp)

            else:
                error = self.manager.createErrorModel(qdist, self.precision, self.exp, self.poly_precision)
                quantized_distribution = self.manager.createBinOperation(qdist, "*+", error, self.poly_precision)
        else:
            self.evaluate(tree.left)
            dist = self.manager.createUnaryOperation(tree.left.root_value[0], tree.root_name, tree.root_name)
            qdist = self.manager.createUnaryOperation(tree.left.root_value[2], tree.root_name, tree.root_name)

            if isPointMassDistr(dist):
                error = ErrorModelPointMass(qdist, self.precision, self.exp)
                quantized_distribution = quantizedPointMass(dist, self.precision, self.exp)
            else:
                error = self.manager.createErrorModel(qdist, self.precision, self.exp, self.poly_precision)
                quantized_distribution = self.manager.createBinOperation(qdist, "*+", error, self.poly_precision)

        # We now populate the triple with distribution, error model, quantized distribution '''
        tree.root_value = [dist, error, quantized_distribution]

    def generate_output_samples(self, amount_time):
        """ Generate sample_nb samples of tree evaluation in the tree's working precision
            :return an array of samples """
        print("Generating Samples...")
        d = np.zeros(1)
        setCurrentContextPrecision(self.precision, self.exp)
        start_time=time.time()
        end_time=0
        while end_time<=amount_time:
            self.resetInit(self.tree)
            d = numpy.append(d, float(printMPFRExactly(self.evaluate_at_sample(self.tree))))
            end_time=time.time()-start_time
        resetContextDefault()
        return d

    def evaluate_at_sample(self, tree):
        """ Sample from the leaf then evaluate tree in the tree's working precision"""
        if tree.left is not None or tree.right is not None:
           if tree.left is not None:
               sample_l = self.evaluate_at_sample(tree.left)
           if tree.right is not None:
               sample_r = self.evaluate_at_sample(tree.right)
           if tree.root_name == "+":
               return gmpy2.add(mpfr(str(sample_l)), mpfr(str(sample_r)))
           elif tree.root_name == "-":
               return gmpy2.sub(mpfr(str(sample_l)), mpfr(str(sample_r)))
           elif tree.root_name == "*":
               return gmpy2.mul(mpfr(str(sample_l)), mpfr(str(sample_r)))
           elif tree.root_name == "/":
               return gmpy2.div(mpfr(str(sample_l)), mpfr(str(sample_r)))
           elif tree.root_name == "exp":
               return gmpy2.exp(mpfr(str(sample_l)))
           elif tree.root_name == "sin":
               return gmpy2.sin(mpfr(str(sample_l)))
           elif tree.root_name == "cos":
               return gmpy2.cos(mpfr(str(sample_l)))
           else:
               print("Operation not supported!")
               exit(-1)
        else:
           sample = tree.root_value[0].getSampleSet(n=1)[0]
           return mpfr(str(sample))

    def generate_error_samples(self, sample_nb):
        """ Generate sample_nb samples of tree evaluation in the tree's working precision
                    :return an array of samples """
        e = np.zeros(sample_nb)
        setCurrentContextPrecision(self.precision, self.exp)
        for i in range(0, sample_nb):
            sample, lp_sample = self.evaluate_error_at_sample(self.tree)
            e[i] = (sample - lp_sample) / (self.eps * sample)
        resetContextDefault()
        return e

    def evaluate_error_at_sample(self, tree):
        """ Sample from the leaf then evaluate tree in the tree's working precision"""
        if tree.left is not None or tree.right is not None:
            if tree.left is not None:
                sample_l, lp_sample_l = self.evaluate_error_at_sample(tree.left)
            if tree.right is not None:
                sample_r, lp_sample_r = self.evaluate_error_at_sample(tree.right)
            if tree.root_name == "+":
                return (sample_l + sample_r), gmpy2.add(mpfr(str(lp_sample_l)), mpfr(str(lp_sample_r)))
            elif tree.root_name == "-":
                return (sample_l - sample_r), gmpy2.sub(mpfr(str(lp_sample_l)), mpfr(str(lp_sample_r)))
            elif tree.root_name == "*":
                return (sample_l * sample_r), gmpy2.mul(mpfr(str(lp_sample_l)), mpfr(str(lp_sample_r)))
            elif tree.root_name == "/":
                return (sample_l / sample_r), gmpy2.div(mpfr(str(lp_sample_l)), mpfr(str(lp_sample_r)))
            else:
                print("Operation not supported!")
                exit(-1)
        else:
            sample = tree.root_value[0].execute().rand()
            return sample, mpfr(str(sample))

    def resetInit(self, tree):
        if tree.left is not None or tree.right is not None:
           if tree.left is not None:
               self.resetInit(tree.left)
           if tree.right is not None:
               self.resetInit(tree.right)
           tree.root_value[0].resetSampleInit()
        else:
           tree.root_value[0].resetSampleInit()

    def collectInfoAboutDistribution(self, f):
        res="Info:\n\n"
        finalDistr=self.tree.root_value[2].execute()
        mode=finalDistr.mode()
        res=res+"Mode of the distribution: " + str(mode) + "\n\n\n"
        gap=abs(self.tree.root_value[2].a-self.tree.root_value[2].b)
        gap=gap/1000.0
        for i in [0.25, 0.5, 0.75, 0.85, 0.95, 0.99, 0.9999]:
            val = 0
            lower = mode
            upper = mode
            while val<i:
                lower=lower-gap
                if lower<self.tree.root_value[2].a:
                    lower=self.tree.root_value[2].a
                upper = upper + gap
                if upper>self.tree.root_value[2].b:
                    upper=self.tree.root_value[2].b
                val=finalDistr.get_piecewise_pdf().integrate(lower,upper)
                if val>=i:
                    res=res+"Range: ["+str(lower)+","+str(upper)+"] contains "+str(i*100)+"% of the distribution.\n\n"
                    break
        res = res + "Range: [" + str(self.tree.root_value[2].a) + "," + str(self.tree.root_value[2].b) + "] contains 100% of the distribution.\n\n"
        print(res)
        f.write(res)
        return


    def plot_range_analysis(self, fileHook, final_time, path, file_name):
        self.resetInit(self.tree)
        r = self.generate_output_samples(final_time)
        self.tree.root_value[2].execute()
        a = self.tree.root_value[2].a
        b = self.tree.root_value[2].b
        # as bins, choose at the intervals between successive pairs of representable numbers between a and b

        tmp_precision = 2
        tmp_exp = self.exp

        for binLen in [1000, 5000, 10000]:
            bins = []

            #if self.precision>11:
            #    tmp_precision=11
            #    tmp_exp = self.exp
            #else:
            #    tmp_precision=self.precision
            #    tmp_exp=self.exp

            while True:
                setCurrentContextPrecision(tmp_precision, tmp_exp)
                f = mpfr(str(a))
                if a < float(printMPFRExactly(f)):
                    f = gmpy2.next_below(f)
                while f < b:
                    bins.append(float(printMPFRExactly(f)))
                    f = gmpy2.next_above(f)
                resetContextDefault()
                if len(bins)>=binLen:
                    break
                else:
                    bins=[]
                    tmp_precision = tmp_precision+1

            if len(bins)==0:
                bins=1

            print("Generating Graphs\n")

            fileHook.write("Picked Precision for histogram. mantissa:"+str(tmp_precision)+", exp:"+str(tmp_exp)+"\n\n")
            fileHook.flush()

            for i in range(0,10,2):
                tmp_filename=file_name+"_Out_"+str(i)+"_Bins_"+str(binLen)
                plt.figure(tmp_filename, figsize=(15,10))
                vals, edges, patches =plt.hist(r, bins, density=True, color="b")
                self.elaborateBinsAndEdges(fileHook, a, b, edges, vals)
                x = np.linspace(a, b, 1000)
                plt.ylim(0, sorted(vals, reverse=True)[i])
                plt.plot(x, abs(self.tree.root_value[2].distribution.get_piecewise_pdf()(x)), linewidth=7, color="red")
                #plotTicks(file_name,"X","g", 2, 500, ticks=[7.979, 16.031], label="FPT: [7.979, 16.031]")
                plotBoundsDistr(tmp_filename, self.tree.root_value[2].distribution)
                #plotTicks(file_name, "|", "g", 6, 600, ticks=[9.0, 15.0], label="99.99% prob. dist.\nin [9.0, 15.0]")
                plt.xlabel('Distribution Range')
                plt.ylabel('PDF')
                plt.title(file_name+"\nmantissa="+str(self.precision)+", exp="+str(self.exp)+"\n")
                plt.legend(fontsize=25)
                #+file_name.replace('./', '')
                plt.savefig(path+file_name+"/"+tmp_filename, dpi = 100)
                plt.close("all")

    def elaborateBinsAndEdges(self, fileHook, a, b, edges, vals):
        #counter=np.count_nonzero(vals==0.0)
        counter=0.0
        tot=0.0
        abs_counter=0
        for ind, val in enumerate(vals):
            gap = abs(edges[ind + 1] - edges[ind])
            if val==0:
                counter=counter+gap
                abs_counter=abs_counter+1
            tot=tot+gap

        fileHook.write("Abs - Empty Bins: "+str(abs_counter)+", out of "+str(len(vals))+ " total bins.\n")
        fileHook.write("Abs - Ratio: " + str(float(abs_counter)/float(len(vals)))+"\n\n")
        fileHook.write("Weighted - Empty Bins: " + str(counter) + ", out of " + str(tot) + " total bins.\n")
        fileHook.write("Weighted - Ratio: " + str(float(counter) / float(tot)) + "\n\n")

    def plot_empirical_error_distribution(self, sample_nb, file_name):
        e = self.generate_error_samples(sample_nb)
        a = math.floor(e.min())
        b = math.ceil(e.max())
        # as bins, choose multiples of 2*eps between a and b
        bins = np.linspace(a, b, (b-a) * 2**(self.precision-1))
        plt.hist(e, bins, density=True)
        plt.savefig("pics/" + file_name)
        plt.close("all")

class quantizedPointMass:

   def __init__(self, wrapperInputDistribution, precision, exp):
       self.wrapperInputDistribution = wrapperInputDistribution
       self.inputdistribution = self.wrapperInputDistribution.execute()
       self.precision = precision
       self.exp = exp
       setCurrentContextPrecision(self.precision, self.exp)
       qValue = printMPFRExactly(mpfr(str(self.inputdistribution.rand(1)[0])))
       resetContextDefault()
       self.name = qValue
       self.sampleInit=True
       self.distribution = ConstDistr(float(qValue))
       self.distribution.get_piecewise_pdf()
       self.a = self.distribution.range_()[0]
       self.b = self.distribution.range_()[-1]

   def execute(self):
       return self.distribution

   def resetSampleInit(self):
       self.sampleInit=True

   def getSampleSet(self, n=100000):
       # it remembers values for future operations
       if self.sampleInit:
           # self.sampleSet = self.distribution.rand(n)
           if n <= 2:
               self.sampleSet = self.distribution.rand(n)
           else:
               self.sampleSet = self.distribution.rand(n - 2)
               self.sampleSet = np.append(self.sampleSet, [self.a, self.b])
           self.sampleInit = False
       return self.sampleSet

tmp_pdf=None
def my_tmp_pdf(t):
    return tmp_pdf(t)

bins=None
n=None
def op(t):
    if isinstance(t, float) or isinstance(t, int) or len(t) == 1:
        if t < min(bins) or t > max(bins):
            return 0.0
        else:
            index_bin=np.digitize(t,bins)
            return n[index_bin]
    else:
        res=np.zeros(len(t))
        tis=t
        for index,ti in enumerate(tis):
            if ti < min(bins) or ti > max(bins):
                res[index] = 0.0
            else:
                index_bin = np.digitize(ti, bins, right=True)
                res[index] = n[index_bin-1]
        return res
    return 0

class BinOpDist:
    """
    Wrapper class for the result of an arithmetic operation on PaCal distributions
    Warning! leftoperand and rightoperant MUST be PaCal distributions
    """
    def __init__(self, leftoperand, operator, rightoperand, poly_precision, regularize=True, convolution=True):
        self.leftoperand = leftoperand
        self.operator = operator
        self.rightoperand = rightoperand

        self.name="("+self.leftoperand.name+str(self.operator)+self.rightoperand.name+")"

        self.poly_precision=poly_precision

        self.regularize = regularize
        self.convolution=convolution

        self.distribution=None
        self.distributionConv = None
        self.distributionSamp = None

        self.sampleInit=True
        self.execute()

    def executeIndependent(self):
        if self.operator == "+":
            self.distributionConv = self.leftoperand.execute() + self.rightoperand.execute()
        elif self.operator == "-":
            self.distributionConv = self.leftoperand.execute() - self.rightoperand.execute()
        elif self.operator == "*":
            self.distributionConv = self.leftoperand.execute() * self.rightoperand.execute()
        elif self.operator == "/":
            self.distributionConv = self.leftoperand.execute() / self.rightoperand.execute()
        # operator to multiply by a relative error
        elif self.operator == "*+":
            self.distributionConv = self.leftoperand.execute() * (1.0 + (self.rightoperand.eps*self.rightoperand.execute()))
        else:
            print("Operation not supported!")
            exit(-1)

        self.distributionConv.get_piecewise_pdf()

        if self.regularize:
            self.distributionConv = chebfunInterpDistr(self.distributionConv, 10)
            self.distributionConv = normalizeDistribution(self.distributionConv)

        self.aConv = self.distributionConv.range_()[0]
        self.bConv = self.distributionConv.range_()[-1]

    def operationDependent(self):
        leftOp = self.leftoperand.getSampleSet()
        rightOp = self.rightoperand.getSampleSet()

        if self.operator == "*+":
            res = np.array(leftOp) * (1 + (self.rightoperand.eps * np.array(rightOp)))
        else:
            res = eval("np.array(leftOp)" + self.operator + "np.array(rightOp)")

        return res

    def executeDependent(self):

        res = self.distributionValues

        bin_nb = int(math.ceil(math.sqrt(len(res))))

        global n, bins
        n, bins, patches = plt.hist(res, bins=bin_nb, density=True)

        breaks=[min(bins), max(bins)]

        global tmp_pdf
        tmp_pdf = chebfun(op, domain=breaks, N=self.poly_precision)

        #global tmp_pdf
        #tmp_pdf = chebfun(lambda t: op(t, bins, n), domain=[min(bins), max(bins)], N=100)

        self.distributionSamp = MyFunDistr(my_tmp_pdf, breakPoints=breaks, interpolated=True)
        self.distributionSamp.init_piecewise_pdf()

        if self.regularize:
            self.distributionSamp = chebfunInterpDistr(self.distributionSamp, 10)
            self.distributionSamp = normalizeDistribution(self.distributionSamp)

        self.aSamp = self.distributionSamp.range_()[0]
        self.bSamp = self.distributionSamp.range_()[-1]

    def execute(self):
        if self.distribution==None:
            if self.convolution:
                self.executeIndependent()
                self.distributionValues = self.operationDependent()
                self.distribution=self.distributionConv
                self.a = self.aConv
                self.b = self.bConv
            else:
                self.distributionValues = self.operationDependent()
                self.executeDependent()
                self.distribution = self.distributionSamp
                self.a = self.aSamp
                self.b = self.bSamp

            self.distribution.get_piecewise_pdf()
        return self.distribution

    def getSampleSet(self,n=100000):
        #it remembers values for future operations
        if self.sampleInit:
            self.execute()
            self.sampleSet  = self.distributionValues
            self.sampleInit = False
        return self.sampleSet

    def resetSampleInit(self):
        self.sampleInit=True

class UnOpDist:
    """
    Wrapper class for the result of unary operation on a PaCal distribution
    """
    def __init__(self, operand, name, operation=None):
        if operation is None:
            self.distribution = operand.execute()
        elif operation is "exp":
            self.distribution = pacal.exp(operand.execute())
            self.distribution.get_piecewise_pdf()
        elif operation is "cos":
            self.distribution = pacal.cos(operand.execute())
            self.distribution.get_piecewise_pdf()
        elif operation is "sin":
            self.distribution = pacal.sin(operand.execute())
            self.distribution.get_piecewise_pdf()
        else:
            print("Unary operation not yet supported")
            exit(-1)

        self.operand = operand
        self.name = name
        self.a = self.distribution.range_()[0]
        self.b = self.distribution.range_()[-1]

    def execute(self):
        return self.distribution

    def resetSampleInit(self):
        self.operand.sampleInit=True

    def getSampleSet(self,n=100000):
        return self.operand.getSampleSet(n)
