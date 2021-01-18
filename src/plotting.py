from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from IntervalArithmeticLibrary import find_max_abs_interval
from setup_utils import output_path
from storage import load_histograms_error_from_disk, \
    load_histograms_range_from_disk, store_histograms_error, store_histograms_range
from evaluation import collectInfoAboutSampling, collectInfoAboutCDFDistributionINV, measureDistances, \
    collectInfoAboutCDFDistributionNaive, collectInfoAboutCDFSampling, collectInfoAboutCDFDistributionPBox

plt.rcParams.update({'font.size': 30})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'legend.frameon': False})
plt.rcParams.update({'legend.handletextpad': 0.1})
plt.rcParams.update({'legend.labelspacing': 0.5})
plt.rcParams.update({'axes.labelpad': 20})
plt.rcParams.update({'legend.loc':'best'})

def plotTicks(figureName, mark, col, lw, s, ticks, label=""):
    if not ticks is None and not None in eval(ticks):
        values_ticks=eval(ticks)
        minVal = values_ticks[0]
        maxVal = values_ticks[1]
        labelMinVal = str('%.1e' % float(minVal))
        labelMaxVal = str("%.1e" % float(maxVal))
        plt.figure(figureName)
        plt.scatter(x=[minVal, maxVal], y=[0, 0], c=col, marker=mark, label="FPTaylor: [" + labelMinVal + "," + labelMaxVal + "]", linewidth=lw, s=s)

def plotConstraints(figureName, mark, col, lw, s, ticks, label=""):
    if not ticks is None and not None in eval(ticks):
        values_ticks=eval(ticks)
        minVal = values_ticks[0]
        maxVal = values_ticks[1]
        labelMinVal = str('%.1e' % float(minVal))
        labelMaxVal = str("%.1e" % float(maxVal))
        plt.figure(figureName)
        plt.scatter(x=[minVal, maxVal], y=[0, 0], c=col, marker=mark, label="PAF 99%: [" + labelMinVal + "," + labelMaxVal + "]", linewidth=lw, s=s)

def plotBoundsDistr(figureName, distribution):
    minVal = distribution.range_()[0]
    maxVal = distribution.range_()[-1]
    labelMinVal = str("%.1e" % distribution.range_()[0])
    labelMaxVal = str("%.1e" % distribution.range_()[-1])
    plt.figure(figureName)
    plt.scatter(x=[minVal, maxVal], y=[0, 0], c='r', marker="|",
                label="PAF: [" + labelMinVal + "," + labelMaxVal + "]", linewidth=6, s=600)


def plot_range_analysis_PDF(final_distribution, loadedGolden, golden_samples, paf_file, file_name, range_fpt):

    print("Generating Graphs Range Analysis PDF\n")

    tmp_filename = file_name + "_range_PDF_Bins_Auto"
    plt.figure(tmp_filename, figsize=(15, 10))

    if loadedGolden:
        vals_golden, edges_golden = load_histograms_range_from_disk(file_name)
        plt.fill_between(edges_golden, np.concatenate(([0], vals_golden)), step="pre", color="darkgoldenrod",
                         label="Golden distribution")
    else:
        vals_golden_10000, edges_golden_10000 = np.histogram(golden_samples, bins=10000, density=True)
        vals_golden, edges_golden, patch_discard = plt.hist(golden_samples, bins='auto', density=True,
                                                             color="darkgoldenrod", label="Golden distribution")
        store_histograms_range(file_name,vals_golden, edges_golden, vals_golden_10000, edges_golden_10000)

    golden_file = open(output_path + file_name + "/golden_pdf.txt", "a+")
    binLenGolden = len(vals_golden)
    title="PDF Range Analysis with Golden with num. bins: " + str(binLenGolden)
    golden_mode, golden_ind = collectInfoAboutSampling(golden_file, vals_golden, edges_golden, title, pdf=True)
    golden_file.close()

    distr_mode = final_distribution.distribution.mode()
    binLenDistr = 1000
    title="PDF Range Analysis with PAF with gaps: " + str(binLenDistr)
    collectInfoAboutCDFDistributionNaive(paf_file, final_distribution, title, distr_mode, binLenDistr)

    #sampling_file = open(output_path + file_name + "/sampling.txt", "a+")
    #vals, edges, patches = plt.hist(r, bins='auto', density=True, color="blue", label="Sampled distribution")
    #binLenSamp = len(vals)
    #title="PDF Range Analysis with Sampling Model with num. bins: " + str(binLenSamp)
    #collectInfoAboutSampling(sampling_file, vals, edges, title, pdf=True)
    #sampling_file.close()

    #title="PDF Measure Distances Range Analysis"
    #measureDistances(final_distribution, paf_file, vals_golden, vals, edges_golden, edges, title)

    golden_max = abs(final_distribution.distribution.get_piecewise_pdf()(golden_mode))
    mode_distr = final_distribution.distribution.mode()
    distr_max = abs(final_distribution.distribution.get_piecewise_pdf()(mode_distr))

    finalMax = max(golden_max, distr_max)

    plt.autoscale(enable=True, axis='both', tight=False)
    plt.ylim(top=2.0 * finalMax)
    #x = np.linspace(a, b, 1000)
    #plt.plot(x, abs(final_distribution.distribution.get_piecewise_pdf()(x)), linewidth=3, color="red")
    final_distribution.distribution.plot(linewidth=3, color="red")
    plotTicks(tmp_filename, "X", "green", 4, 500, ticks=range_fpt, label="FPT: " + str(range_fpt))
    plotBoundsDistr(tmp_filename, final_distribution.distribution)
    plt.xlabel('Distribution Range')
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    plt.ylabel('PDF')
    plt.title(file_name + " - Range Analysis\n")
    plt.legend(fontsize=25)
    plt.savefig(output_path + file_name + "/" + tmp_filename, dpi=100)
    plt.clf()
    plt.close()

def plotCDF(edges, vals, normalize, **kwargs):
    if normalize:
        tmp_vals = vals / sum(vals)
    else:
        tmp_vals = vals
    cdf_tmp=np.insert(np.cumsum(tmp_vals), 0, 0.0, axis=0)
    plt.plot(edges, cdf_tmp , **kwargs)
    return cdf_tmp, edges

def plotCDFdiscretization(insiders):
    plot_boxing(insiders)

    evaluation_points=set()
    for pbox in insiders:
        evaluation_points.add(Decimal(pbox.interval.lower))
        evaluation_points.add(Decimal(pbox.interval.upper))
    evaluation_points = sorted(evaluation_points)

    edge_cdf = []
    val_cdf_low = []
    val_cdf_up = []

    res_ub = Decimal("-inf")  # Interval("0.0", "0.0", True, True, digits_for_cdf)
    res_lb = Decimal("+inf")  # Interval("0.0", "0.0", True, True, digits_for_cdf)

    for ev_point in evaluation_points:
        if ev_point==Decimal(insiders[0].interval.lower):
            edge_cdf.append(float(ev_point))
            val_cdf_low.append(0)
            val_cdf_up.append(0)
        elif ev_point == Decimal(insiders[-1].interval.upper):
            edge_cdf.append(float(ev_point))
            val_cdf_low.append(1)
            val_cdf_up.append(1)
        else:
            for inside in insiders:
                if Decimal(inside.interval.lower) < ev_point < Decimal(inside.interval.upper):
                    res_ub=max(res_ub, Decimal(inside.cdf_up)) #.addition(Interval(inside.cdf_low,inside.cdf_up,True,True, digits_for_cdf))
                    res_lb=min(res_lb, Decimal(inside.cdf_low))
                elif Decimal(inside.interval.lower) == ev_point and inside.interval.include_lower:
                    res_ub=max(res_ub, Decimal(inside.cdf_up)) #res_ub.addition(Interval(inside.cdf_low,inside.cdf_up,True,True, digits_for_cdf))
                    res_lb=min(res_lb, Decimal(inside.cdf_low))
                elif Decimal(inside.interval.upper) == ev_point and inside.interval.include_upper:
                    res_ub=max(res_ub, Decimal(inside.cdf_up)) #res_ub.addition(Interval(inside.cdf_low,inside.cdf_up,True,True, digits_for_cdf))
                    res_lb=min(res_lb, Decimal(inside.cdf_low))
            edge_cdf.append(float(ev_point))
            val_cdf_low.append(float(res_lb))
            val_cdf_up.append(float(res_ub))
            res_ub = Decimal("-inf")
            res_lb = Decimal("+inf")
    plt.plot(edge_cdf, val_cdf_low, 'X', c="black")
    plt.plot(edge_cdf, val_cdf_up, 'o', c="blue")
    plt.show(block=False)

def plot_boxing(ret_list):
    ax = plt.gca()
    plt.show(block=False)
    for val in ret_list:
        ax.add_patch(Rectangle((float(val.interval.lower), float(val.cdf_low)),
                               float(val.interval.upper)-float(val.interval.lower),
                               float(val.cdf_up)-float(val.cdf_low), color='black',
                               fill = False, label=("Boxing" if val==ret_list[0] else "")))
    plt.legend()
    return


def collectInfoAboutErrorWithConstraints(fileHook, error_results):
    fileHook.write("\n#### Error values with Gelpia ####\n\n")
    for value in error_results:
        fileHook.write(value+":"+str(error_results[value])+"\n")
    fileHook.write("#############\n\n")
    return

def plot_range_analysis_CDF(T, loadedGolden, samples_golden, fileHook, file_name, range_fpt):
    final_distribution=T.final_quantized_distr

    print("Generating Graphs Range Analysis CDF\n")

    tmp_filename = file_name + "_range_CDF_Bins_Auto"
    plt.figure(tmp_filename, figsize=(15, 10))

    if loadedGolden:
        notnorm_vals_golden, notnorm_edges_golden = load_histograms_range_from_disk(file_name)
        vals_golden, edges_golden = plotCDF(notnorm_edges_golden, notnorm_vals_golden, normalize=True,
                                                 color="darkgoldenrod", linewidth=3, label="Golden distribution")
    else:
        not_norm_vals_golden_10000, not_norm_edges_golden_10000 = np.histogram(samples_golden, bins=10000, density=True)
        not_norm_vals_golden, not_norm_edges_golden = np.histogram(samples_golden, bins='auto', density=True)
        store_histograms_range(file_name, not_norm_vals_golden,not_norm_edges_golden,not_norm_vals_golden_10000,not_norm_edges_golden_10000)
        vals_golden, edges_golden = plotCDF(not_norm_edges_golden, not_norm_vals_golden, normalize=True,
                                                 color="darkgoldenrod", linewidth=3, label="Golden distribution")

    golden_file = open(output_path + file_name + "/golden_range.txt", "a+")
    binLenGolden = len(vals_golden)
    title="CDF Range Analysis with Golden with num. bins: " + str(binLenGolden)
    collectInfoAboutSampling(golden_file, vals_golden, edges_golden, title, pdf=False, golden_mode_index=0)
    golden_file.close()

    #title = "CDF Range Analysis with PAF using INV CDF "
    #collectInfoAboutCDFDistributionINV(fileHook, final_distribution, title)

    title = "CDF Range Analysis with PAF using PBox Discretization"
    collectInfoAboutCDFDistributionPBox(fileHook, final_distribution, title)
    collectInfoAboutErrorWithConstraints(fileHook, T.error_results)
    #sampling_file = open(output_path + file_name + "/sampling.txt", "a+")
    #notnorm_vals, notnorm_edges = np.histogram(samples_short, bins='auto', density=True)
    #vals, edges = plotCDF(notnorm_edges, notnorm_vals, normalize=True, color="blue", label="Sampled distribution", linewidth=3)
    #binLenSamp = len(vals)
    #title="CDF Range Analysis with Sampling with num. bins: " + str(binLenSamp)
    #collectInfoAboutCDFSampling(sampling_file, notnorm_vals, edges, title)
    #sampling_file.close()

    #title="CDF Measure Distances Range Analysis"
    #measureDistances(final_distribution, fileHook, vals_golden, vals, edges_golden, edges, title, pdf=False)

    plt.autoscale(enable=True, axis='both', tight=False)
    plt.ylim(bottom=-0.05, top=1.1)
    #x = np.linspace(a, b, 1000)
    #plt.plot(x, abs(final_distribution.distribution.get_piecewise_cdf()(x)), linewidth=3, color="red")
    #final_distribution.distribution.get_piecewise_cdf().plot(xmin=a, xmax=b, linewidth=3, color="red")
    #plotBoundsDistr(tmp_filename, final_distribution.distribution)

    plotCDFdiscretization(final_distribution.discretization.intervals)

    plotTicks(tmp_filename, "X", "green", 4, 500, ticks=range_fpt, label="FPT: " + str(range_fpt))
    plt.xlabel('Distribution Range')
    plt.ylabel('CDF')
    plt.title(file_name + " - Range Analysis")
    plt.legend(fontsize=25)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    plt.savefig(output_path + file_name + "/" + tmp_filename, dpi=100)
    plt.clf()
    plt.close()


def plot_error_analysis_PDF(abs_err, loadedGolden, abs_err_samples, abs_err_golden, summary_file, file_name, abs_fpt, rel_fpt):

    print("Generating Graphs Error Analysis PDF\n")

    tmp_name = file_name + "_abs_error_PDF_Bins_Auto"
    plt.figure(tmp_name, figsize=(15, 10))

    if loadedGolden:
        vals_golden, edges_golden = load_histograms_error_from_disk(file_name)
        plt.fill_between(edges_golden, np.concatenate(([0], vals_golden)), step="pre", color="darkgoldenrod",
                         label="Golden distribution")
    else:
        vals_golden_10000, edges_golden_10000 = np.histogram(abs_err_golden, bins=10000, density=True)
        vals_golden, edges_golden, patch_discard = plt.hist(abs_err_golden, bins='auto', density=True,
                                                    color="darkgoldenrod", label="Golden distribution")
        store_histograms_error(file_name,vals_golden, edges_golden, vals_golden_10000, edges_golden_10000)

    golden_file = open(output_path + file_name + "/golden.txt", "a+")
    binLenGolden = len(vals_golden)
    title="PDF Error Analysis with Golden distribution with num. bins: " + str(binLenGolden)
    golden_mode, golden_ind = collectInfoAboutSampling(golden_file, vals_golden, edges_golden, title, pdf=True)
    golden_file.close()

    binLenDistr = 1000
    title = "PDF Error Analysis with PAF with gap: " + str(binLenDistr)
    collectInfoAboutCDFDistributionNaive(summary_file, abs_err, title, abs_err.a, binLenDistr)

    sampling_file = open(output_path + file_name + "/sampling.txt", "a+")
    vals, edges, patches = plt.hist(abs_err_samples, bins='auto', density=True, color="blue", label="Sampling model")
    binLenSamp = len(vals)
    title="PDF Error Analysis with Sampling Model with num. bins: " + str(binLenSamp)
    collectInfoAboutSampling(sampling_file, vals, edges, title, pdf=True)
    sampling_file.close()

    title="PDF Measure Distances Abs Error\n"
    measureDistances(abs_err, summary_file, vals_golden, vals, edges_golden, edges, title)

    golden_max = abs(abs_err.execute().get_piecewise_pdf()(golden_mode))
    mode_distr = abs_err.execute().mode()
    distr_max = abs(abs_err.execute().get_piecewise_pdf()(mode_distr))

    finalMax = max(golden_max, distr_max)

    plt.autoscale(enable=True, axis='both', tight=False)
    plt.ylim(top=2.0 * finalMax)
    x = np.linspace(abs_err.a, abs_err.b, 1000)
    plt.plot(x, abs(abs_err.distribution.get_piecewise_pdf()(x)), linewidth=5, color="red")
    plotTicks(tmp_name, "X", "green", 4, 500, ticks="[0.0, " + str(abs_fpt) + "]", label="FPT: " + str(abs_fpt))
    plotBoundsDistr(tmp_name, abs_err.distribution)
    plt.xlabel('Distribution Range')
    plt.ylabel('PDF')
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    plt.title(tmp_name)
    plt.legend(fontsize=25)
    plt.savefig(output_path + file_name + "/" + tmp_name)
    plt.clf()
    plt.close()


def plot_abs_error_analysis_CDF(tree, loadedGolden, abs_err_golden, summary_file, file_name, abs_fpt, rel_fpt):
    #abs_err=tree.abs_err_distr

    print("Generating Graphs Abs Error Analysis CDF\n")

    tmp_name = file_name + "_abs_error_CDF_Bins_Auto"
    plt.figure(tmp_name, figsize=(15, 10))

    if loadedGolden:
        notnorm_vals_golden, notnorm_edges_golden = load_histograms_error_from_disk(file_name)
        vals_golden, edges_golden = plotCDF(notnorm_edges_golden, notnorm_vals_golden, normalize=True,
                                                 color="darkgoldenrod", linewidth=3, label="Golden distribution")
    else:
        not_norm_vals_golden_10000, not_norm_edges_golden_10000 = np.histogram(abs_err_golden, bins=10000, density=True)
        not_norm_vals_golden, not_norm_edges_golden = np.histogram(abs_err_golden, bins='auto', density=True)
        store_histograms_error(file_name, not_norm_vals_golden,not_norm_edges_golden,not_norm_vals_golden_10000,not_norm_edges_golden_10000)
        vals_golden, edges_golden = plotCDF(not_norm_edges_golden, not_norm_vals_golden, normalize=True,
                                                 color="darkgoldenrod", linewidth=3, label="Golden distribution")


    golden_file = open(output_path + file_name + "/golden_abs_error.txt", "a+")
    binLenGolden = len(vals_golden)
    title="CDF ABS Error Analysis with Golden with num. bins: " + str(binLenGolden)
    collectInfoAboutSampling(golden_file, vals_golden, edges_golden, title, pdf=False, golden_mode_index=0)
    golden_file.close()

    #title = "CDF ABS Error Analysis with PAF using INV CDF "
    #collectInfoAboutCDFDistributionINV(summary_file, abs_err, title)
    #title = "CDF ABS Error Analysis with PAF using PBox Discretization"
    #collectInfoAboutCDFDistributionPBox(summary_file, abs_err, title)


    #sampling_file = open(output_path + file_name + "/sampling.txt", "a+")
    #not_norm_vals, not_norm_edges = np.histogram(abs_err_samples, bins='auto', density=True)
    #vals, edges = plotCDF(not_norm_edges, not_norm_vals, normalize=True, linewidth=3, color="blue",label="Sampled distribution")
    #binLenSamp = len(vals)
    #title="CDF Error Analysis with Sampling model with num. bins: " + str(binLenSamp)
    #collectInfoAboutSampling(sampling_file, vals, edges, title, pdf=False, golden_mode_index=0)
    #sampling_file.close()
    #title="CDF Measure Distances Error Analysis"
    #measureDistances(abs_err, summary_file, vals_golden, vals, edges_golden, edges, title, pdf=False)

    error_bounds=tree.error_results
    plt.autoscale(enable=True, axis='both', tight=False)
    plt.ylim(bottom=-0.05, top=1.1)

    #x = np.linspace(abs_err.a, abs_err.b, 1000)
    #plt.plot(x, abs(abs_err.distribution.get_piecewise_cdf()(x)), linewidth=3, color="red")
    #abs_err.distribution.get_piecewise_cdf().plot(xmin=abs_err.a, xmax=abs_err.b, linewidth=3, color="red")
    #plotBoundsDistr(tmp_name, abs_err.distribution)
    #tree.lower_error_affine.get_piecewise_cdf().plot(xmin=0, xmax=tree.lower_error_affine.range_()[-1], linewidth=3, color="green")
    #tree.upper_error_affine.get_piecewise_cdf().plot(xmin=0, xmax=tree.upper_error_affine.range_()[-1], linewidth=3, color="green")
    #plotCDFdiscretization(abs_err.discretization.intervals)

    plotTicks(tmp_name, "X", "green", 4, 500, ticks="[0.0, " + str(abs_fpt) + "]", label="FPT: " + str(abs_fpt))
    paf_99=error_bounds["0.99"]
    plotConstraints(tmp_name, "+", "red", 4, 500, ticks="[0.0, " + str(find_max_abs_interval(paf_99)) + "]")

    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    plt.title(tmp_name)
    plt.xlabel('ABS Error Distribution')
    plt.ylabel('CDF')
    plt.legend(fontsize=25)
    plt.savefig(output_path + file_name + "/" + tmp_name)
    plt.clf()
    plt.close()

def plot_error_analysis_CDF(tree, loadedGolden, abs_err_golden, summary_file, file_name, abs_fpt, rel_fpt):
    abs_err=tree.err_distr

    print("Generating Graphs Error Analysis CDF\n")

    tmp_name = file_name + "_error_CDF_Bins_Auto"
    plt.figure(tmp_name, figsize=(15, 10))

    if loadedGolden:
        notnorm_vals_golden, notnorm_edges_golden = load_histograms_error_from_disk(file_name)
        vals_golden, edges_golden = plotCDF(notnorm_edges_golden, notnorm_vals_golden, normalize=True,
                                                 color="darkgoldenrod", linewidth=3, label="Golden distribution")
    else:
        not_norm_vals_golden_10000, not_norm_edges_golden_10000 = np.histogram(abs_err_golden, bins=10000, density=True)
        not_norm_vals_golden, not_norm_edges_golden = np.histogram(abs_err_golden, bins='auto', density=True)
        store_histograms_error(file_name, not_norm_vals_golden,not_norm_edges_golden,not_norm_vals_golden_10000,not_norm_edges_golden_10000)
        vals_golden, edges_golden = plotCDF(not_norm_edges_golden, not_norm_vals_golden, normalize=True,
                                                 color="darkgoldenrod", linewidth=3, label="Golden distribution")


    golden_file = open(output_path + file_name + "/golden_error.txt", "a+")
    binLenGolden = len(vals_golden)
    title="CDF Error Analysis with Golden with num. bins: " + str(binLenGolden)
    collectInfoAboutSampling(golden_file, vals_golden, edges_golden, title, pdf=False, golden_mode_index=0)
    golden_file.close()

    title = "CDF Error Analysis with PAF using INV CDF "
    collectInfoAboutCDFDistributionINV(summary_file, abs_err, title)

    title = "CDF Error Analysis with PAF using PBox Discretization"
    collectInfoAboutCDFDistributionPBox(summary_file, abs_err, title)

    #sampling_file = open(output_path + file_name + "/sampling.txt", "a+")
    #not_norm_vals, not_norm_edges = np.histogram(abs_err_samples, bins='auto', density=True)
    #vals, edges = plotCDF(not_norm_edges, not_norm_vals, normalize=True, linewidth=3, color="blue",label="Sampled distribution")
    #binLenSamp = len(vals)
    #title="CDF Error Analysis with Sampling model with num. bins: " + str(binLenSamp)
    #collectInfoAboutSampling(sampling_file, vals, edges, title, pdf=False, golden_mode_index=0)
    #sampling_file.close()

    #title="CDF Measure Distances Error Analysis"
    #measureDistances(abs_err, summary_file, vals_golden, vals, edges_golden, edges, title, pdf=False)

    plt.autoscale(enable=True, axis='both', tight=False)
    plt.ylim(bottom=-0.05, top=1.1)
    #x = np.linspace(abs_err.a, abs_err.b, 1000)
    #plt.plot(x, abs(abs_err.distribution.get_piecewise_cdf()(x)), linewidth=3, color="red")
    #abs_err.distribution.get_piecewise_cdf().plot(xmin=abs_err.a, xmax=abs_err.b, linewidth=3, color="red")
    #plotBoundsDistr(tmp_name, abs_err.distribution)
    #tree.lower_error_affine.get_piecewise_cdf().plot(xmin=0, xmax=tree.lower_error_affine.range_()[-1], linewidth=3, color="green")
    #tree.upper_error_affine.get_piecewise_cdf().plot(xmin=0, xmax=tree.upper_error_affine.range_()[-1], linewidth=3, color="green")

    plotCDFdiscretization(abs_err.discretization.intervals)

    plotTicks(tmp_name, "X", "green", 4, 500, ticks="[0.0, " + str(abs_fpt) + "]", label="FPT: " + str(abs_fpt))

    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    plt.title(tmp_name)
    plt.xlabel('Error Distribution')
    plt.ylabel('CDF')
    plt.legend(fontsize=25)
    plt.savefig(output_path + file_name + "/" + tmp_name)
    plt.clf()
    plt.close()

def plot_operation(edge_cdf,val_cdf_low,val_cdf_up):
    plt.figure()
    plt.plot([float(a) for a in edge_cdf], [float(a) for a in val_cdf_low], 'o', markersize=15, c="red", label="lower_bound_SMT")
    plt.plot([float(a) for a in edge_cdf], [float(a) for a in val_cdf_up], 'x', markersize=15, c="purple", label="upper_bound_SMT")
    plt.ticklabel_format(axis='both', style='sci')
    plt.legend()
