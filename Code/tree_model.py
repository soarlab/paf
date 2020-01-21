from error_model import *
from regularizer import *
import time
from scipy.stats import *
import os.path
from operations import *

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

class DistributionsManager:
    def __init__(self, samples_dep_op):
        self.samples_dep_op = samples_dep_op
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

    def createBinOperation(self, leftoperand, operator, rightoperand, poly_precision,  regularize=True, convolution=True):
        name="("+leftoperand.name+str(operator)+rightoperand.name+")"
        if name in self.distrdictionary:
            return self.distrdictionary[name]
        else:
            tmp=BinOpDist(leftoperand, operator, rightoperand, poly_precision, self.samples_dep_op, regularize, convolution)
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

    def __init__(self, my_yacc, precision, exp, poly_precision, samples_dep_op, initialize=True):
        self.initialize = initialize
        self.precision = precision
        self.exp = exp
        self.poly_precision = poly_precision
        # Copy structure of the tree from my_yacc
        self.tree = copy_tree(my_yacc.expression)
        # Evaluate tree
        self.eps = 2 ** (-self.precision)
        self.samples_dep_op=samples_dep_op
        self.manager=DistributionsManager(self.samples_dep_op)
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
        return d[1:]

    def generate_error_samples(self, sample_time, name, loadIfExists=False):
        """ Generate sample_nb samples of tree evaluation in the tree's working precision
                    :return an array of samples """
        print("Generating Samples...")

        if loadIfExists and os.path.exists("./storage/"+name+"/"):
            return True, [], [], []

        rel_err = []#np.zeros(1)
        abs_err = []#np.zeros(1)
        values  = []#np.zeros(1)

        setCurrentContextPrecision(self.precision, self.exp)

        start_time = time.time()
        end_time=0
        while end_time<=sample_time:
            self.resetInit(self.tree)
            sample, lp_sample = self.evaluate_error_at_sample(self.tree)
            values.append(sample)
            tmp_abs = abs(float(printMPFRExactly(lp_sample)) - sample)
            abs_err.append(tmp_abs)
            rel_err.append(tmp_abs/sample)
            #rel_err = numpy.append(rel_err, abs((float(printMPFRExactly(lp_sample)) - sample)/sample)) # self.eps *
            #abs_err = numpy.append(abs_err, tmp_abs) # self.eps *
            end_time=time.time()-start_time
        resetContextDefault()
        return False, np.asarray(values), np.asarray(abs_err), np.asarray(rel_err)

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
            sample = tree.root_value[0].getSampleSet(n=1)[0]
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

    def collectInfoAboutDistribution(self, f, finalDistr_wrapper, name, distr_mode, bin_len):
        res="###### Info about "+name+"#######:\n\n"

        res=res+"Mode of the distribution: " + str(distr_mode) + "\n\n\n"
        gap=abs(finalDistr_wrapper.a-finalDistr_wrapper.b)
        gap=gap/float(bin_len)
        for i in [0.25, 0.5, 0.75, 0.85, 0.95, 0.99, 0.9999]:
            val = 0
            lower = distr_mode
            upper = distr_mode
            while val<i:
                lower=lower-gap
                if lower<finalDistr_wrapper.a:
                    lower=finalDistr_wrapper.a
                upper = upper + gap
                if upper>finalDistr_wrapper.b:
                    upper=finalDistr_wrapper.b
                val=finalDistr_wrapper.execute().get_piecewise_pdf().integrate(lower,upper)
                if val>=i:
                    res=res+"Range: ["+str(lower)+","+str(upper)+"] contains "+str(i*100)+"% of the distribution.\n\n"
                    break
        res = res + "Range: [" + str(finalDistr_wrapper.a) + "," + str(finalDistr_wrapper.b) + "] contains 100% of the distribution.\n\n"
        res = res+"###########################################\n\n"
        #print(res)
        f.write(res)
        return

    def collectInfoAboutSampling(self, f, vals, edges, name, golden_mode=None, golden_ind=None):
        res="###### Info about "+name+"#######:\n\n"
        if golden_mode is None:
            ind = vals.argmax()
            mode = edges[ind]
        else:
            ind = golden_ind
            mode = golden_mode
        res=res+"Mode of the sampling distribution: " + str(mode) + "\n\n\n"
        tot=sum(vals)
        for i in [0.25, 0.5, 0.75, 0.85, 0.95, 0.99, 0.9999]:
            val = vals[ind]
            lower = ind
            upper = ind+1
            while (val/tot) < i:
                lower = lower - 1
                if lower < 0:
                    lower = 0
                upper = upper + 1
                if upper > len(edges)-1:
                    upper = len(edges)-1
                val = sum(vals[lower:upper])
            res = res + "Range: [" + str(edges[lower]) + "," + str(edges[upper]) + "] contains " + str(i * 100) + "% of the distribution.\n\n"
        res = res + "Range: [" + str(edges[0]) + "," + str(edges[-1]) + "] contains 100% of the distribution.\n\n"
        res = res+"###########################################\n\n"
        #print(res)
        f.write(res)
        return mode, ind

    def measureDistrVsGoldenEdges(self, distr, edges_golden):
        vals=[]
        distr_pdf=distr.distribution.get_piecewise_pdf()
        for ind, edge in enumerate(edges_golden[:-1]):
            if edge>=distr.a and edge<=distr.b:
                vals.append(abs(distr_pdf(edge)))
            else:
                vals.append(0.0)
        return vals

    def my_KL_entropy(self, p, q):
        return scipy.stats.entropy(p, q)

    def getValueHist(self, edges, vals, x):
        if x < min(edges) or x >= max(edges):
            return 0.0
        else:
            index_bin=np.digitize(x,edges,right=False)
            return abs(vals[index_bin-1])

    def measureDistances(self, distr_wrapper, fileHook, vals_PM_orig, vals_golden, vals_orig, edges_PM_orig, edges_golden, edges_orig, introStr):

        #if not (len(vals_PM_orig)==len(vals_golden) and len(vals_golden)==len(vals_orig)):
        #    print("Failure in histograms!")
        #    exit(-1)

        vals_DistrPM = np.asarray(self.measureDistrVsGoldenEdges(distr_wrapper, edges_golden))

        vals_PM=[]
        vals=[]
        for edge_golden in edges_golden[:-1]:
            vals_PM.append(self.getValueHist(edges_PM_orig, vals_PM_orig, edge_golden))
            vals.append(self.getValueHist(edges_orig, vals_orig, edge_golden))

        vals_PM=np.asarray(vals_PM)
        vals=np.asarray(vals)
        #self.outputEdgesVals(fileHook, "BinLen: " + str(len(vals_golden)) + ", FP_or_real: " + str(fp_or_real) + "\n\n",
        #                     edges_golden, vals_DistrPM)

        var_distance_golden_DistrPM = np.max(np.absolute(vals_golden - vals_DistrPM))
        var_distance_golden_PM=np.max(np.absolute(vals_golden-vals_PM))
        var_distance_golden_sampling=np.max(np.absolute(vals_golden-vals))

        avg_var_distance_golden_DistrPM=np.average(np.absolute(vals_golden-vals_DistrPM))
        avg_var_distance_golden_PM=np.average(np.absolute(vals_golden-vals_PM))
        avg_var_distance_golden_sampling=np.average(np.absolute(vals_golden-vals))

        KL_distance_golden_DistrPM=self.my_KL_entropy(vals_golden, vals_DistrPM)
        KL_distance_golden_PM=self.my_KL_entropy(vals_golden, vals_PM)
        KL_distance_golden_sampling=self.my_KL_entropy(vals_golden, vals)

        WSS_distance_golden_DistrPM=scipy.stats.wasserstein_distance(vals_golden, vals_DistrPM)
        WSS_distance_golden_PM=scipy.stats.wasserstein_distance(vals_golden, vals_PM)
        WSS_distance_golden_sampling=scipy.stats.wasserstein_distance(vals_golden, vals)

        fileHook.write("##### DISTANCE MEASURES ######\n\n")
        fileHook.write(introStr+"\n")

        fileHook.write("Variational Distance - Golden -> DistrPM : " + str(var_distance_golden_DistrPM) + "\n")
        fileHook.write("Variational Distance - Golden -> PM : "+str(var_distance_golden_PM)+"\n")
        fileHook.write("Variational Distance - Golden -> Sampling : "+str(var_distance_golden_sampling)+"\n")

        fileHook.write("AVG Variational Distance - Golden -> DistrPM : " + str(avg_var_distance_golden_DistrPM) + "\n")
        fileHook.write("AVG Variational Distance - Golden -> PM : " + str(avg_var_distance_golden_PM) + "\n")
        fileHook.write("AVG Variational Distance - Golden -> Sampling : " + str(avg_var_distance_golden_sampling) + "\n")

        fileHook.write("KL Distance - Golden -> DistrPM : " + str(KL_distance_golden_DistrPM) + "\n")
        fileHook.write("KL Distance - Golden -> PM : " + str(KL_distance_golden_PM) + "\n")
        fileHook.write("KL Distance - Golden -> Sampling : " + str(KL_distance_golden_sampling) + "\n")

        fileHook.write("WSS Distance - Golden -> DistrPM : " + str(WSS_distance_golden_DistrPM) + "\n")
        fileHook.write("WSS Distance - Golden -> PM : " + str(WSS_distance_golden_PM) + "\n")
        fileHook.write("WSS Distance - Golden -> Sampling : " + str(WSS_distance_golden_sampling) + "\n")

        fileHook.write("##################################")
        fileHook.flush()
        return


    def plot_range_analysis(self, loadedGolden, r, golden_samples, fileHook, path, file_name, range_fpt):

        self.tree.root_value[2].execute()
        a = self.tree.root_value[2].a
        b = self.tree.root_value[2].b

        # expand to fptaylor probably
        # as bins, choose at the intervals between successive pairs of representable numbers between a and b
        tmp_precision = 2
        tmp_exp = self.exp

        fp_or_real=False


        '''
        for binLen in [50, 100, 500, 1000, 5000, 10000]:
            bins = []
            if fp_or_real:
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

                fileHook.write("Picked Precision for histogram. mantissa:"+str(tmp_precision)+", exp:"+str(tmp_exp)+"\n\n")
                fileHook.flush()

            else:
                bins=binLen
        '''

        print("Generating Graphs Range Analysis\n")

        tmp_filename=file_name+"range__Bins_Auto"#+str(binLen)

        plt.figure(tmp_filename, figsize=(15,10))
        golden_file=open(path + file_name + "/golden.txt","a+")

        if loadedGolden:
            data_auto = np.load("./storage/"+file_name+"/"+file_name+"_range_auto.npz")
            data_tt = np.load("./storage/"+file_name+"/"+file_name+"_range_10000.npz")
            vals_golden, edges_golden = data_auto["vals_golden"], data_auto["edges_golden"]
            plt.fill_between(edges_golden, np.concatenate(([0], vals_golden)), step="pre", color="black", label="Golden model")
            data_auto.close()
            data_tt.close()
        else:
            vals_golden_tt, edges_golden_tt, patches_golden_tt = plt.hist(golden_samples, bins=10000, alpha=0.0, density=True, color="black")
            vals_golden, edges_golden, patches_golden = plt.hist(golden_samples, bins='auto', density=True, color="black", label="Golden model")
            if not os.path.exists("./storage/"+file_name+"/"):
                os.makedirs("./storage/"+file_name+"/")
            np.savez("./storage/"+file_name+"/"+file_name+"_range_10000.npz", vals_golden=vals_golden_tt, edges_golden=edges_golden_tt)
            np.savez("./storage/"+file_name+"/"+file_name+"_range_auto.npz", vals_golden=vals_golden, edges_golden=edges_golden)

        binLenGolden=len(vals_golden)

        golden_mode, golden_ind=self.collectInfoAboutSampling(golden_file,vals_golden,edges_golden,"Golden with num. bins: "+str(binLenGolden))
        #self.outputEdgesVals(golden_file,"BinLen: "+str(binLen)+", FP_or_real: "+str(fp_or_real)+"\n\n",edges_golden,vals_golden)
        golden_file.close()

        distr_mode = self.tree.root_value[2].distribution.mode()
        binLenDistr=1000
        self.collectInfoAboutDistribution(fileHook, self.tree.root_value[2], "Range Analysis on Round(distr) with "+str(binLenDistr)+" bins", distr_mode, binLenDistr)

        pm_file=open(path + file_name + "/pm.txt","a+")
        vals_PM, edges_PM, patches_PM =plt.hist(self.tree.root_value[2].distributionValues, bins='auto', density=True, alpha=0.0, color="red")
        #vals_PM, edges_PM, patches_PM =plt.hist(self.tree.root_value[2].distributionValues, edges_golden, density=True, alpha=0.0, color="red")
        binLenPM = len(vals_PM)
        pm_mode,pm_ind=self.collectInfoAboutSampling(pm_file,vals_PM,edges_PM,"PM with num. bins: "+str(binLenPM))#, golden_mode=golden_mode, golden_ind=golden_ind)
        #self.outputEdgesVals(pm_file,"BinLen: "+str(binLen)+", FP_or_real: "+str(fp_or_real)+"\n\n",edges_PM,vals_PM)
        pm_file.close()

        sampling_file=open(path + file_name + "/sampling.txt","a+")
        vals, edges, patches =plt.hist(r, bins='auto', alpha=0.5, density=True, color="blue", label="Sampling model")
        #vals, edges, patches =plt.hist(r, edges_golden, alpha=0.5, density=True, color="blue", label="Sampling model")
        binLenSamp=len(vals)
        self.collectInfoAboutSampling(sampling_file,vals,edges,"Sampling with num. bins: "+str(binLenSamp))#, golden_mode=golden_mode, golden_ind=golden_ind)
        #self.outputEdgesVals(sampling_file, "BinLen: "+str(binLenSamp)+", FP_or_real: "+str(fp_or_real)+"\n\n", edges, vals)
        sampling_file.close()

        self.measureDistances(self.tree.root_value[2], fileHook, vals_PM, vals_golden, vals, edges_PM, edges_golden, edges, "Measure Distances Range Analysis")

        golden_max = abs(self.tree.root_value[2].distribution.get_piecewise_pdf()(golden_mode))
        pm_max = abs(self.tree.root_value[2].distribution.get_piecewise_pdf()(pm_mode))
        mode_distr = self.tree.root_value[2].distribution.mode()
        distr_max = abs(self.tree.root_value[2].distribution.get_piecewise_pdf()(mode_distr))

        finalMax=max(golden_max, pm_max, distr_max)

        plt.autoscale(enable=True, axis='both', tight=False)
        plt.ylim(top=2.0*finalMax)
        x = np.linspace(a, b, 1000)
        plt.plot(x, abs(self.tree.root_value[2].distribution.get_piecewise_pdf()(x)), linewidth=5, color="red")
        plotTicks(tmp_filename,"X","green", 4, 500, ticks=range_fpt, label="FPT: "+str(range_fpt))
        plotBoundsDistr(tmp_filename, self.tree.root_value[2].distribution)
        #plotTicks(file_name, "|", "g", 6, 600, ticks=[9.0, 15.0], label="99.99% prob. dist.\nin [9.0, 15.0]")
        plt.xlabel('Distribution Range')
        plt.ylabel('PDF')
        plt.title(file_name+" - Range Analysis"+"\nprec="+str(self.precision)+", exp="+str(self.exp)+"\n")
        plt.legend(fontsize=25)
        #+file_name.replace('./', '')
        plt.savefig(path+file_name+"/"+tmp_filename, dpi = 100)
        plt.close("all")

    def outputEdgesVals(self, file_hook, string_name, edges, vals):
        file_hook.write(string_name)
        for ind, val in enumerate(vals):
            file_hook.write("["+str(edges[ind])+","+str(edges[ind+1])+"] -> "+str(val)+"\n")
        file_hook.write("\n\n")
        file_hook.flush()

    def elaborateBinsAndEdges(self, fileHook, edges, vals, name):
        #counter=np.count_nonzero(vals==0.0)
        counter=0.0
        tot=0.0
        abs_counter=0
        fileHook.write("##### Info about: "+str(name)+"#######\n\n\n")

        for ind, val in enumerate(vals):
            gap = abs(edges[ind + 1] - edges[ind])
            if val==0:
                counter=counter+gap
                abs_counter=abs_counter+1
                fileHook.write("Bin ["+str(edges[ind])+","+str(edges[ind+1])+"] is empty.\n")
            tot=tot+gap

        for ind, val in enumerate(vals):
            if not val==0:
                fileHook.write("Leftmost bin not empty: ["+str(edges[ind])+","+str(edges[ind+1])+"].\n")
                break

        for ind, val in reversed(list(enumerate(vals))):
            if not val==0:
                fileHook.write("Rightmost bin not empty: ["+str(edges[ind])+","+str(edges[ind+1])+"]\n")
                break

        fileHook.write("Abs - Empty Bins: "+str(abs_counter)+", out of "+str(len(vals))+ " total bins.\n")
        fileHook.write("Abs - Ratio: " + str(float(abs_counter)/float(len(vals)))+"\n\n")
        fileHook.write("Weighted - Empty Bins: " + str(counter) + ", out of " + str(tot) + " total bins.\n")
        fileHook.write("Weighted - Ratio: " + str(float(counter) / float(tot)) + "\n\n")
        fileHook.write("########################\n\n")

    def plot_empirical_error_distribution(self, loadedGolden, abs_err_samples, abs_err_golden, summary_file, benchmarks_path, file_name, abs_fpt, rel_fpt):

        abs_err = UnOpDist(BinOpDist(self.tree.root_value[2],"-", self.tree.root_value[0], 500, 250000, regularize=True, convolution=False), "abs_err", "abs")
        #rel_err = UnOpDist(BinOpDist(BinOpDist(self.tree.root_value[2],"-", self.tree.root_value[0], 100, regularize=True, convolution=False), "/", self.tree.root_value[0], 100, regularize=True, convolution=False), "rel_err", "abs")

        abs_err.execute().get_piecewise_pdf()

        abs_a = abs_err_samples.min()
        abs_b = abs_err_samples.max()

        '''
        tmp_name="Rel_Error_Bins_"+str(n_bin)
        plt.figure(tmp_name, figsize=(15, 10))
        bins = np.linspace(rel_a, rel_b, n_bin)
        vals, edges, patches = plt.hist(rel_err_samples, bins, density=True)
        rel_err.distribution.plot(linewidth=4, color="red")
        if not rel_fpt is None:
            plotTicks(tmp_name,"X","green", 2, 500, ticks=[0, rel_fpt], label="FPT: "+str(rel_fpt))
            plt.xlim(0, max(float(rel_fpt), rel_err.b))
        else:
            plt.xlim(0, float(rel_err.b))
        plotBoundsDistr(tmp_name, rel_err.distribution)
        plt.title(tmp_name)
        plt.legend(fontsize=25)
        plt.savefig(benchmarks_path+file_name+"/"+tmp_name)
        self.elaborateBinsAndEdges(summary_file, edges, vals, "Relative Error Distribution (samples)")
        '''

        print("Generating Graphs Error Analysis\n")

        tmp_name=file_name+"_Abs_Error_Bins_Auto"
        plt.figure(tmp_name, figsize=(15, 10))

        distr_mode = abs_err.execute().mode()
        self.collectInfoAboutDistribution(summary_file, abs_err, "Abs Error Distribution", distr_mode, 1000)

        if loadedGolden:
            data_auto = np.load("./storage/"+file_name+"/"+file_name+"_error_auto.npz")
            data_tt = np.load("./storage/"+file_name+"/"+file_name+"_error_10000.npz")
            vals_golden, edges_golden = data_auto["vals_golden"], data_auto["edges_golden"]
            plt.fill_between(edges_golden, np.concatenate(([0], vals_golden)), step="pre", color="black", label="Golden model")
            data_auto.close()
            data_tt.close()
        else:
            vals_golden_tt, edges_golden_tt, patches_golden_tt = plt.hist(abs_err_golden, bins=10000, alpha=0.0, density=True, color="black")
            vals_golden, edges_golden, patches_golden = plt.hist(abs_err_golden, bins='auto', density=True, color="black", label="Golden model")
            if not os.path.exists("./storage/"+file_name+"/"):
                os.makedirs("./storage/"+file_name+"/")
            np.savez("./storage/"+file_name+"/"+file_name+"_error_10000.npz", vals_golden=vals_golden_tt, edges_golden=edges_golden_tt)
            np.savez("./storage/"+file_name+"/"+file_name+"_error_auto.npz", vals_golden=vals_golden, edges_golden=edges_golden)

        binLenGolden = len(vals_golden)
        golden_mode, golden_ind = self.collectInfoAboutSampling(summary_file, vals_golden, edges_golden, "Golden with num. bins: " + str(binLenGolden))

        vals, edges, patches = plt.hist(abs_err_samples, bins='auto', alpha=0.5, density=True, color="blue", label="Sampling model")
        binLenSamp = len(vals)
        self.collectInfoAboutSampling(summary_file, vals, edges, "Sampling with num. bins: " + str(binLenSamp))

        vals_PM, edges_PM, patches_PM = plt.hist(np.absolute(abs_err.operand.distributionValues), bins='auto', density=True, alpha=0.0, color="red")
        binLenPM = len(vals_PM)
        pm_mode, pm_ind = self.collectInfoAboutSampling(summary_file, vals_PM, edges_PM, "PM with num. bins: " + str(binLenPM))

        golden_max = abs(abs_err.execute().get_piecewise_pdf()(golden_mode))
        pm_max = abs(abs_err.execute().get_piecewise_pdf()(pm_mode))
        mode_distr = abs_err.execute().mode()
        distr_max = abs(abs_err.execute().get_piecewise_pdf()(mode_distr))

        finalMax = max(golden_max, pm_max, distr_max)

        plt.autoscale(enable=True, axis='both', tight=False)
        plt.ylim(top=2.0 * finalMax)

        x = np.linspace(abs_err.a, abs_err.b, 1000)
        plt.plot(x, abs(abs_err.distribution.get_piecewise_pdf()(x)), linewidth=5, color="red")

        plotTicks(tmp_name, "X", "green", 4, 500, ticks="[0.0, "+str(abs_fpt)+"]", label="FPT: " + str(abs_fpt))

        plotBoundsDistr(tmp_name, abs_err.distribution)

        plt.title(tmp_name)
        plt.legend(fontsize=25)
        plt.savefig(benchmarks_path+file_name+"/"+tmp_name)

        self.measureDistances(abs_err, summary_file, vals_PM, vals_golden, vals, edges_PM, edges_golden, edges,
                              "Measure Distances Abs error")

        plt.close("all")
