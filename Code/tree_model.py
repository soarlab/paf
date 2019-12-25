from model import *
from error_model import *
from regularizer import *

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

    def createErrorModel(self, wrapDist, precision, exp, pol_prec):
        if wrapDist.name in self.errordictionary:
            return self.errordictionary[wrapDist.name]
        else:
            tmp=ErrorModel(wrapDist, precision, exp, pol_prec)
            self.errordictionary[wrapDist.name]=tmp
            return tmp

    def createBinOperation(self, leftoperand, operator, rightoperand, regularize=True, convolution=True):
        name="("+leftoperand.name+str(operator)+rightoperand.name+")"
        if name in self.distrdictionary:
            return self.distrdictionary[name]
        else:
            tmp=BinOpDist(leftoperand, operator, rightoperand, regularize, convolution)
            self.distrdictionary[name]=tmp
            return tmp

    def createUnaryOperation(self, operand, name):
        if name in self.distrdictionary:
            return  self.distrdictionary[name]
        else:
            tmp=UnOpDist(operand, name)
            self.distrdictionary[name]=tmp
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
                    quantized_distribution = self.manager.createBinOperation(dist, "*+", error)
            # Else we leave the leaf distribution unchanged
            else:
                error = 0
                quantized_distribution = dist

        # If not at a leaf we need to get the distribution and quantized distributions of the children nodes.
        # Then, check the operation. For each operation the template is the same:
        # dist will be the non-quantized operation the non-quantized children nodes
        # qdist will be the non-quantized operation on the quantized children nodes
        # quantized_distribution will be the quantized operation on the quantized children nodes

        else:

            self.evaluate(tree.left)
            self.evaluate(tree.right)

            dist  = self.manager.createBinOperation(tree.left.root_value[0], tree.root_name, tree.right.root_value[0], convolution=tree.convolution)
            qdist = self.manager.createBinOperation(tree.left.root_value[2], tree.root_name, tree.right.root_value[2], convolution=tree.convolution)
            error = self.manager.createErrorModel(qdist, self.precision, self.exp, self.poly_precision)

            quantized_distribution = self.manager.createBinOperation(qdist, "*+", error)

        # We now populate the triple with distribution, error model, quantized distribution '''
        tree.root_value = [dist, error, quantized_distribution]

    def generate_output_samples(self, sample_nb):
        """ Generate sample_nb samples of tree evaluation in the tree's working precision
            :return an array of samples """
        d = np.zeros(sample_nb)
        setCurrentContextPrecision(self.precision, self.exp)
        for i in range(0, sample_nb):
            self.resetInit(self.tree)
            d[i] = float(printMPFRExactly(self.evaluate_at_sample(self.tree)))
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
           tree.root_value[0].sampleInit=True
        else:
           tree.root_value[0].sampleInit=True

    def plot_range_analysis(self, sample_nb, file_name):
        self.resetInit(self.tree)
        r = self.generate_output_samples(sample_nb)
        self.tree.root_value[2].execute()
        a = self.tree.root_value[2].a
        b = self.tree.root_value[2].b
        # as bins, choose at the intervals between successive pairs of representable numbers between a and b
        bins = []
        setCurrentContextPrecision(self.precision, self.exp)
        f = mpfr(str(a))
        if a < float(printMPFRExactly(f)):
            f = gmpy2.next_below(f)
        while f < b:
            bins.append(float(printMPFRExactly(f)))
            f = gmpy2.next_above(f)
        resetContextDefault()
        plt.figure(file_name, figsize=(15,10))
        plt.hist(r, bins, density=True, color="b")
        x = np.linspace(a, b, 1000)
        plt.plot(x, abs(self.tree.root_value[2].distribution.get_piecewise_pdf()(x)), linewidth=7, color="red")
        #plotTicks(file_name,"X","g", 2, 500, ticks=[7.979, 16.031], label="FPT: [7.979, 16.031]")
        plotBoundsDistr(file_name, self.tree.root_value[2].distribution)
        #plotTicks(file_name, "|", "g", 6, 600, ticks=[9.0, 15.0], label="99.99% prob. dist.\nin [9.0, 15.0]")
        plt.xlabel('Distribution Range')
        plt.ylabel('PDF')
        plt.title(file_name+"\nmantissa="+str(self.precision)+", exp="+str(self.exp)+"\n")
        plt.legend(fontsize=25)
        plt.savefig("./pics/"+file_name, dpi = 100)
        plt.close("all")

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
       qValue = printMPFRExactly(mpfr(str(self.inputdistribution)))
       resetContextDefault()
       self.distribution = ConstDistr(float(qValue))
       self.distribution.init_piecewise_pdf()

   def execute(self):
       return self.distribution

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
    def __init__(self, leftoperand, operator, rightoperand, regularize=True, convolution=True):
        self.leftoperand = leftoperand
        self.operator = operator
        self.rightoperand = rightoperand

        self.name="("+self.leftoperand.name+str(self.operator)+self.rightoperand.name+")"

        self.regularize = regularize
        self.convolution=convolution

        self.distribution=None
        self.distributionConv = None
        self.distributionSamp = None

        self.sampleInit=True
        self.execute()

    def executeIndipendent(self):
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

        self.distributionConv.init_piecewise_pdf()

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
        tmp_pdf = chebfun(op, domain=breaks, N=100)

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
                self.executeIndipendent()
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


class UnOpDist:
    """
    Wrapper class for the result of unary operation on a PaCal distribution
    """
    def __init__(self, operand, name, operation=None):
        if operation is None:
            self.operand=operand
            self.name=name
            self.sampleInit=True
            self.distribution = operand.execute()
            self.a = self.distribution.range_()[0]
            self.b = self.distribution.range_()[-1]
        else:
            print("Unary operation not yet supported")
            exit(-1)

    def execute(self):
        return self.distribution

    def getSampleSet(self,n=100000):
        #it remembers values for future operations
        if self.sampleInit:
            #self.sampleSet = self.distribution.rand(n)
            if n<=2:
                self.sampleSet = self.distribution.rand(n)
            else:
                self.sampleSet  = self.distribution.rand(n-2)
                self.sampleSet  = np.append(self.sampleSet, [self.a, self.b])
            self.sampleInit = False
        return self.sampleSet
