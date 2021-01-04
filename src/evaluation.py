from decimal import Decimal

import numpy as np
import statsmodels.stats
import scipy.stats

def measureDistances(distr_wrapper, fileHook, vals_golden, vals_orig, edges_golden, edges_orig, introStr, pdf=True):

    vals_DistrPM = np.asarray(measureDistrVsGoldenEdges(distr_wrapper, edges_golden, pdf))

    vals = []
    if pdf:
        edges_golden=edges_golden[:-1]

    for edge_golden in edges_golden:
        vals.append(getValueHist(edges_orig, vals_orig, edge_golden, pdf))
    vals = np.asarray(vals)

    var_distance_golden_DistrPM = np.max(np.absolute(vals_golden - vals_DistrPM))
    var_distance_golden_sampling = np.max(np.absolute(vals_golden - vals))
    avg_var_distance_golden_DistrPM = np.average(np.absolute(vals_golden - vals_DistrPM))
    avg_var_distance_golden_sampling = np.average(np.absolute(vals_golden - vals))
    KL_distance_golden_DistrPM = my_KL_entropy(vals_golden, vals_DistrPM)
    KL_distance_golden_sampling = my_KL_entropy(vals_golden, vals)
    WSS_distance_golden_DistrPM = scipy.stats.wasserstein_distance(vals_golden, vals_DistrPM)
    WSS_distance_golden_sampling = scipy.stats.wasserstein_distance(vals_golden, vals)

    fileHook.write("##### DISTANCE MEASURES ######\n\n")
    fileHook.write(introStr + "\n")
    fileHook.write("Variational Distance - Golden -> paf : " + str(var_distance_golden_DistrPM) + "\n")
    fileHook.write("Variational Distance - Golden -> Sampling : " + str(var_distance_golden_sampling) + "\n")
    fileHook.write("AVG Variational Distance - Golden -> paf : " + str(avg_var_distance_golden_DistrPM) + "\n")
    fileHook.write("AVG Variational Distance - Golden -> Sampling : " + str(avg_var_distance_golden_sampling) + "\n")
    fileHook.write("KL Distance - Golden -> paf : " + str(KL_distance_golden_DistrPM) + "\n")
    fileHook.write("KL Distance - Golden -> Sampling : " + str(KL_distance_golden_sampling) + "\n")
    fileHook.write("WSS Distance - Golden -> paf : " + str(WSS_distance_golden_DistrPM) + "\n")
    fileHook.write("WSS Distance - Golden -> Sampling : " + str(WSS_distance_golden_sampling) + "\n")
    fileHook.write("##################################\n\n")
    fileHook.flush()
    return

def outputEdgesVals(file_hook, string_name, edges, vals):
    file_hook.write(string_name)
    for ind, val in enumerate(vals):
        file_hook.write("[" + str(edges[ind]) + "," + str(edges[ind + 1]) + "] -> " + str(val) + "\n")
    file_hook.write("\n\n")
    file_hook.flush()

def elaborateBinsAndEdges(fileHook, edges, vals, name):
    counter = 0.0
    tot = 0.0
    abs_counter = 0
    fileHook.write("##### Info about: " + str(name) + "#######\n\n\n")

    for ind, val in enumerate(vals):
        gap = abs(edges[ind + 1] - edges[ind])
        if val == 0:
            counter = counter + gap
            abs_counter = abs_counter + 1
            fileHook.write("Bin [" + str(edges[ind]) + "," + str(edges[ind + 1]) + "] is empty.\n")
        tot = tot + gap

    for ind, val in enumerate(vals):
        if not val == 0:
            fileHook.write("Leftmost bin not empty: [" + str(edges[ind]) + "," + str(edges[ind + 1]) + "].\n")
            break

    for ind, val in reversed(list(enumerate(vals))):
        if not val == 0:
            fileHook.write("Rightmost bin not empty: [" + str(edges[ind]) + "," + str(edges[ind + 1]) + "]\n")
            break

    fileHook.write("Abs - Empty Bins: " + str(abs_counter) + ", out of " + str(len(vals)) + " total bins.\n")
    fileHook.write("Abs - Ratio: " + str(float(abs_counter) / float(len(vals))) + "\n\n")
    fileHook.write("Weighted - Empty Bins: " + str(counter) + ", out of " + str(tot) + " total bins.\n")
    fileHook.write("Weighted - Ratio: " + str(float(counter) / float(tot)) + "\n\n")
    fileHook.write("########################\n\n")

def collectInfoAboutCDFDistributionINV(f, finalDistr_wrapper, name):
    res="###### Info about "+name+"#######:\n\n"
    res=res+"Starting range analysis from: "+str(finalDistr_wrapper.a)+" \n\n\n"
    probs = [0.25, 0.5, 0.75, 0.85, 0.99, 0.999]
    try:
        inv_fun = finalDistr_wrapper.execute().get_piecewise_invcdf(use_interpolated=False)
        for i in probs:
            val = inv_fun(i)
            res = res + "Range: ["+str(finalDistr_wrapper.a)+"," + str(val) + "] contains " + str(i * 100) + "% of the distribution.\n\n"
    except:
        res = res + "Problem with INV CDF\n\n"
        f.write(res)
        f.flush()
        return collectInfoAboutCDFDistributionNaive(f, finalDistr_wrapper, name, finalDistr_wrapper.a, 1000)
    res=res+"Range: ["+str(finalDistr_wrapper.a)+","+str(finalDistr_wrapper.b)+"] contains "+str(100)+"% of the distribution.\n\n"
    res = res+"###########################################\n\n"
    f.write(res)
    return

def collectInfoAboutCDFDistributionPBox(f, finalDistr_wrapper, name):
    res="###### Info about "+name+"#######:\n\n"
    discretization=finalDistr_wrapper.discretization
    res=res+"Starting range analysis from: "+str(discretization.intervals[0].interval.lower)+" \n\n\n"
    probs = ["0.25", "0.5", "0.75", "0.85", "0.98", "0.99", "0.999"]

    upper_discretization = sorted(discretization.intervals,
                                  key=lambda x: Decimal(x.cdf_up), reverse=False)
    lower_discretization = sorted(discretization.intervals,
                                  key=lambda x: Decimal(x.cdf_low), reverse=False)

    for prob in probs:
        for pbox in upper_discretization:
            if Decimal(pbox.cdf_up) >= Decimal(prob):
                res = res + "Range: [" + str(discretization.intervals[0].interval.lower) + "," + str(pbox.interval.upper) \
                                    + "] contains less than " + str(Decimal(prob) * 100) + "% of the distribution (with 100% prob).\n\n"
                break

    for prob in probs:
        for pbox in lower_discretization:
            if Decimal(pbox.cdf_low) >= Decimal(prob):
                res = res + "Range: [" + str(discretization.intervals[0].interval.lower) + "," + str(pbox.interval.upper) \
                                    + "] contains at least " + str(Decimal(prob) * 100) + "% of the distribution  (with 100% prob).\n\n"
                break
    res=res+"Range: ["+str(discretization.intervals[0].interval.lower)+","+str(discretization.intervals[-1].interval.upper)+"] contains 100% of the distribution.\n\n"
    res = res+"###########################################\n\n"
    f.write(res)
    return

def collectInfoAboutCDFDistributionNaive(f, finalDistr_wrapper, name, distr_mode, bin_len):
    res="###### Info about "+name+"#######:\n\n"
    res=res+"Starting range analysis from: " + str(distr_mode) + "\n\n\n"
    gap=abs(finalDistr_wrapper.a-finalDistr_wrapper.b)
    gap=gap/float(bin_len)
    for i in [0.25, 0.5, 0.75, 0.85, 0.95, 0.99, 0.9999]:
        val = 0
        lower = distr_mode
        upper = distr_mode
        lower_limit=False
        upper_limit=False
        while val<i:
            lower=lower-gap
            if lower<finalDistr_wrapper.a:
                lower=finalDistr_wrapper.a
                lower_limit=True
            upper = upper + gap
            if upper>finalDistr_wrapper.b:
                upper=finalDistr_wrapper.b
                upper_limit=True
            val=finalDistr_wrapper.execute().get_piecewise_pdf().integrate(lower,upper)
            if val>=i:
                res=res+"Range: ["+str(lower)+","+str(upper)+"] contains "+str(i*100)+"% of the distribution.\n\n"
                break
            if lower_limit and upper_limit:
                res = res + "Range: [" + str(lower) + "," + str(upper) + "] contains " + str(i * 100) + "% of the distribution.\n\n"
                break

    res = res + "Range: [" + str(finalDistr_wrapper.a) + "," + str(finalDistr_wrapper.b) + "] contains 100% of the distribution.\n\n"
    res = res+"###########################################\n\n"
    f.write(res)
    return

def computeAreas(edges,vals):
    area=np.zeros(len(vals))
    for ind_edge, edge in enumerate(edges[:-1]):
        area[ind_edge] = vals[ind_edge] * abs(edge - edges[ind_edge + 1])
    return area

def collectInfoAboutCDFSampling(f, vals, edges, name):
    res="###### Info about "+name+"#######:\n\n"
    res=res+"Starting from value 0.0\n\n\n"
    area = computeAreas(edges, vals)
    for i in [0.25, 0.5, 0.75, 0.85, 0.95, 0.99, 0.9999, 1]:
        if i == 0.0:
            ret_val=edges[0]
        elif i == 1.0:
            ret_val=edges[-1]
        else:
            # given a measure of probability returns the value of x
            cum_area = np.cumsum(area)
            index = np.digitize(i, cum_area, right=False)
            new_q = i - sum(area[0:index])
            ret_val = (new_q / vals[index]) + edges[index]
        res = res + "Range: [0.0," + str(ret_val) + "] contains " + str(i * 100) + "% of the distribution.\n\n"
    res = res+"###########################################\n\n\n"
    #print(res)
    f.write(res)
    return

def collectInfoAboutSampling(f, vals, edges, name, pdf, golden_mode_index=None):
    res="###### Info about "+name+"#######:\n\n"
    if golden_mode_index is None:
        ind = vals.argmax()
        mode = edges[ind]
    else:
        ind = golden_mode_index
        mode = edges[ind]
    res=res+"Starting from value " + str(mode) + "\n\n\n"
    if pdf:
        tot=sum(vals)
        for i in [0.25, 0.5, 0.75, 0.85, 0.95, 0.99, 0.9999]:
            val = vals[ind]
            lower = ind
            upper = ind+1
            lower_limit = False
            upper_limit = False
            while (val/tot) < i:
                lower = lower - 1
                if lower < 0:
                    lower = 0
                    lower_limit=True
                upper = upper + 1
                if upper > len(edges)-1:
                    upper = len(edges)-1
                    upper_limit=True
                val = sum(vals[lower:upper])
                if lower_limit and upper_limit:
                    break
            res = res + "Range: [" + str(edges[lower]) + "," + str(edges[upper]) + "] contains " + str(i * 100) + "% of the distribution.\n\n"
    else:
        tot = vals[-1]
        for i in [0.25, 0.5, 0.75, 0.85, 0.95, 0.99, 0.9999]:
            val = vals[0]
            lower = 0
            upper = -1
            while (val / tot) < i:
                upper = upper + 1
                val = vals[upper]
                if upper==(len(vals)-1):
                    upper=upper-1
                    break
            res = res + "Range: [" + str(edges[lower]) + "," + str(edges[upper+1]) + "] contains " + str(i * 100) + "% of the distribution.\n\n"
    res = res + "Range: [" + str(edges[0]) + "," + str(edges[-1]) + "] contains 100% of the distribution.\n\n"
    res = res+"###########################################\n\n\n"
    #print(res)
    f.write(res)
    return mode, ind

'''Evaluate the PAF distributions at the edges given by golden model histogram'''
def measureDistrVsGoldenEdges(distr, edges_golden, pdf=True):
    vals=[]
    if pdf:
        distr_fun = distr.distribution.get_piecewise_pdf()
        for ind, edge in enumerate(edges_golden[:-1]):
            if edge >= distr.a and edge <= distr.b:
                vals.append(abs(distr_fun(edge)))
            else:
                vals.append(0.0)
        return vals
    else:
        distr_fun = distr.distribution.get_piecewise_cdf()
        for ind, edge in enumerate(edges_golden):
            if edge <= distr.a:
                vals.append(0.0)
            elif edge >= distr.b:
                vals.append(1.0)
            else:
                vals.append(abs(distr_fun(edge)))
        return vals

def my_KL_entropy(p, q):
    return scipy.stats.entropy(p, q)

def getValueHist(edges, vals, x, pdf):
    if pdf:
        if x <= min(edges) or x >= max(edges):
            return 0.0
        else:
            index_bin=np.digitize(x,edges,right=False)
            return abs(vals[index_bin-1])
    else:
        if x <= min(edges):
            return 0.0
        elif x >= max(edges):
            return 1.0
        else:
            index_bin = np.digitize(x, edges, right=False)
            return abs(vals[index_bin - 1])