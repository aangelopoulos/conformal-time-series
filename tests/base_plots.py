import os, sys, inspect
from itertools import groupby
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from core.utils import plot_coverage_size, plot_sets, plot_vertical_spread, moving_average
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle as pkl
import seaborn as sns
import pdb

def desaturate_color(color, amount=0.5, saturation=None):
    """
    Desaturates the given color by multiplying (1-saturation) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    if saturation is not None:
        return colorsys.hls_to_rgb(c[0], c[1], saturation)
    else:
        return colorsys.hls_to_rgb(c[0], c[1], 1 - amount * (1 - c[2]))

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

if __name__ == "__main__":
    # Open file
    results_filename = sys.argv[1]
    dataset_name = results_filename.split('.')[-2].split('/')[-1]
    metrics_folder = "./metrics/" + dataset_name + "/"
    os.makedirs(metrics_folder, exist_ok=True)
    plots_folder = "./plots/" + dataset_name + "/"
    os.makedirs(plots_folder, exist_ok=True)
    with open(results_filename, 'rb') as handle:
        results = pkl.load(handle)

    # Set style
    cmap_lines = sns.color_palette("husl", len(list(results.keys())))
    sns.set_theme(context="notebook", palette=cmap_lines, style="white", font_scale=4)
    sns.set_style({'axes.spines.right': False, 'axes.spines.top': False})

    # Get alpha and scores
    alpha = results["alpha"]
    scores = results["scores"]
    T_burnin = results["T_burnin"]
    real_data = results["real_data"]
    multiple_series = results["multiple_series"]
    quantiles_given = results["quantiles_given"]
    if real_data:
        forecasts = results["forecasts"]
        data = results["data"]
        del results["forecasts"]
        del results["data"]
        try:
            log = results["log"]
            del results["log"]
        except:
            pass
    del results["alpha"]
    del results["scores"]
    del results["T_burnin"]
    del results["real_data"]
    del results["multiple_series"]
    del results["quantiles_given"]
    keys_to_delete = ["pgd+sat", "unconstrained", "multiplicative", "mirror", "pgd"]
    for key in keys_to_delete:
        del results[key]

    plot_weights_every = T_burnin // 5

    coverages = {}
    nrows = 0
    for key in results.keys():
        try:
            coverages[key] = (results[key]["q"][T_burnin+1:] >= scores[T_burnin+1:]).astype(int).mean()
        except:
            coverages[key] = { lr : (results[key][lr]["q"][T_burnin+1:] >= scores[T_burnin+1:]).astype(int).mean() for lr in list(results[key].keys()) }
            nrows = max(nrows, len(list(results[key].keys())))
    print(coverages)

    # Plot coverage
    linewidth = 2
    transparency = 0.7

    ncols = len(list(results.keys()))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize = (ncols*10.1, nrows*6.4))
    i = 0
    xlabels_nonscores = range(T_burnin+1, scores.shape[0])
    for key in results.keys():
        color = cmap_lines[i]
        if key == "trail":
            axs[0,i].plot(xlabels_nonscores, moving_average((results[key]["q"] >= scores).astype(int))[T_burnin+1:], label=f"cvg={100*coverages[key]:.1f}%", linewidth=linewidth, color=color, alpha=transparency)
            axs[0,i].legend(handlelength=0.0,handletextpad=-0.1)
            axs[0,i].set_title(key)
        else:
            j = 0
            for lr in results[key].keys():
                axs[j,i].plot(xlabels_nonscores, moving_average((results[key][lr]["q"] >= scores).astype(int))[T_burnin+1:], label=f"lr={lr}, cvg={100*coverages[key][lr]:.1f}%", linewidth=linewidth, color=color, alpha=transparency)
                axs[j,i].legend(handlelength=0.0,handletextpad=-0.1)
                j = j + 1
            axs[0,i].set_title(key)
        i = i + 1
    fig.supxlabel('time')
    fig.supylabel('coverage')
    plt.tight_layout(pad=0.05)
    plt.subplots_adjust(left=0.05, bottom=0.07)
    plt.savefig(plots_folder + "coverage.pdf")

    # Size plots!
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols+1, sharex=True, sharey=True, figsize = ((ncols + 1)*10.1, nrows*6.4))
    # Find limits
    largest_finite_value = scores.max()
    smallest_finite_value = scores.min()
    i = 1
    for key in results.keys():
        if key == "trail":
            quantiles = results[key]["q"][T_burnin+1:]
            largest_finite_value = max(largest_finite_value, quantiles[quantiles < np.infty].max())
            smallest_finite_value = min(smallest_finite_value, quantiles[quantiles > -np.infty].min())
        else:
            j = 0
            for lr in results[key].keys():
                quantiles = results[key][lr]["q"][T_burnin+1:]
                largest_finite_value = max(largest_finite_value, quantiles[quantiles < np.infty].max())
                smallest_finite_value = min(smallest_finite_value, quantiles[quantiles > -np.infty].min())
                j = j + 1
        i = i + 1
    # Make plots
    i = 1
    for key in results.keys():
        color = cmap_lines[i-1]
        if key == "trail":
            quantiles = np.clip(results[key]["q"][T_burnin+1:], smallest_finite_value, largest_finite_value)
            axs[0,i].plot(xlabels_nonscores, quantiles, linewidth=linewidth, color=color, alpha=transparency)
            axs[0,i].set_title(key)
        else:
            j = 0
            for lr in results[key].keys():
                quantiles = np.clip(results[key][lr]["q"][T_burnin+1:], smallest_finite_value, largest_finite_value)
                largest_finite_value = max(largest_finite_value, quantiles[quantiles < np.infty].max())
                smallest_finite_value = min(smallest_finite_value, quantiles[quantiles > -np.infty].min())
                axs[j,i].plot(xlabels_nonscores, quantiles, linewidth=linewidth, color=color, alpha=transparency, label=f"lr={lr}")
                axs[j,i].legend(handlelength=0.0,handletextpad=-0.1)
                j = j + 1
            axs[0,i].set_title(key)
        i = i + 1
    axs[0,0].plot(scores,linewidth=linewidth,alpha=transparency,color=cmap_lines[i])
    axs[0,0].set_title("scores")
    #plt.ylim([smallest_finite_value, largest_finite_value])
    plt.ylim([scores.min()*0.9, scores.max()*1.1])
    fig.supxlabel('time')
    fig.supylabel(r'$\hat{q}$')
    plt.tight_layout(pad=0.05)
    plt.subplots_adjust(left=0.05, bottom=0.07)
    plt.savefig(plots_folder + "size.pdf")

    # Weight plots
    #sns.set_theme(context="notebook", palette=cmap_lines, style="white", font_scale=2)
    #sns.set_style({'axes.spines.right': False, 'axes.spines.top': False})
    #cmap = mpl.colormaps['viridis']
    #non_weighted_methods = ["trail", "aci", "quantile", "arima", "arima+quantile", "pid"]
    #non_weighted_methods = [ name for name in non_weighted_methods if name in results.keys() ]
    #fig, axs = plt.subplots(nrows=nrows, ncols=ncols-len(non_weighted_methods), sharex=True, sharey=False, figsize = ((ncols-len(non_weighted_methods))*8, nrows*4.8))
    #i = 0
    #for key in results.keys():
    #    color = cmap_lines[i]
    #    if key in non_weighted_methods:
    #        continue
    #    else:
    #        j = 0
    #        for lr in results[key].keys():
    #            for t in range(T_burnin+1,scores.shape[0],plot_weights_every):
    #                if len(axs.shape) == 2:
    #                    axs[j,i].plot(results[key][lr]["weights"][t,:], linewidth=linewidth, color=cmap(t/scores.shape[0]), alpha=transparency)
    #                    if t == (T_burnin + 1):
    #                        axs[j,i].plot([],[],label=f"lr={lr}")
    #                        axs[j,i].legend(handlelength=0.0,handletextpad=-0.1)
    #                else:
    #                    axs[j].plot(results[key][lr]["weights"][t,:], linewidth=linewidth, color=cmap(t/scores.shape[0]), alpha=transparency)
    #                    if t == (T_burnin + 1):
    #                        axs[j].plot([],[],label=f"lr={lr}")
    #                        axs[j].legend(handlelength=0.0,handletextpad=-0.1)
    #            j = j + 1
    #        if len(axs.shape) == 2:
    #            axs[0,i].set_title(key)
    #        else:
    #            axs[0].set_title(key)
    #    i = i + 1
    #fig.supxlabel('weight index')
    #fig.supylabel(r'$w$')
    #plt.tight_layout()

    #norm = mpl.colors.Normalize(vmin=0, vmax=scores.shape[0])
    #im = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    #fig.colorbar(im, ax=axs.ravel().tolist())
    ##fig.add_subplot(111)
    ##divider = make_axes_locatable(plt.gca())
    ##cax = divider.append_axes("right", size="5%", pad=0.05)

    ##norm = mpl.colors.Normalize(vmin=0, vmax=scores.shape[0])

    ##cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
    ##                                norm=norm,
    ##                                orientation='vertical')
    ##cb1.set_label('time')
    #plt.savefig(plots_folder + f"weights.pdf", bbox_inches="tight")

    # Plot the actual sequence

    if real_data:
        sns.set_theme(context="notebook", palette=cmap_lines, style="white", font_scale=4)
        sns.set_style({'axes.spines.right': False, 'axes.spines.top': False})
        multi_forecast = len(forecasts.shape) > 1
        forecasts = forecasts[T_burnin+1:]
        data = data[T_burnin+1:]
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols+1, sharex=True, sharey=True, figsize = ((ncols + 1)*10.1, nrows*6.4))
        # Find limits
        largest_finite_value = max(10*data.max(), 10*forecasts.max())
        smallest_finite_value = min(0.1*data.min(), 0.1*forecasts.min())
        # Make plots
        i = 1
        for key in results.keys():
            color = lighten_color(desaturate_color(cmap_lines[i-1], saturation=0.3), 0.5)
            if key == "trail":
                quantiles = np.clip(results[key]["q"][T_burnin+1:], smallest_finite_value, largest_finite_value)

                if multi_forecast:
                    sets = [forecasts[:,0] - quantiles, forecasts[:,-1] + quantiles]
                else:
                    sets = [forecasts - quantiles, forecasts + quantiles]

                results[key]["sets"] = sets
                cvds = (sets[0] <= data) & (sets[1] >= data)
                # { lr : (results[key][lr]["q"][T_burnin+1:] >= scores[T_burnin+1:]).astype(int).mean() for lr in list(results[key].keys()) }
                axs[0,i].plot(np.arange(data.shape[0]), data, color='black', alpha=0.2)
                idx_miscovered = np.where(1-cvds)[0]
                axs[0,i].fill_between(np.arange(data.shape[0]), sets[0], sets[1], color=color);
                axs[0,i].scatter(idx_miscovered, data[idx_miscovered], color='#FF000044', s=10)
                axs[0,i].set_title(key)
            else:
                j = 0
                for lr in results[key].keys():
                    quantiles = np.clip(results[key][lr]["q"][T_burnin+1:], smallest_finite_value, largest_finite_value)
                    if multi_forecast:
                        sets = [forecasts[:,0] - quantiles, forecasts[:,-1] + quantiles]
                    else:
                        sets = [forecasts - quantiles, forecasts + quantiles]
                    results[key][lr]["sets"] = sets
                    cvds = (sets[0] <= data) & (sets[1] >= data)
                    axs[j,i].plot(np.arange(data.shape[0]), data, color='black', alpha=0.2)
                    idx_miscovered = np.where(1-cvds)[0]
                    axs[j,i].fill_between(np.arange(data.shape[0]), sets[0], sets[1], color=color, alpha=transparency, label=f"lr={lr}")
                    axs[j,i].scatter(idx_miscovered, data[idx_miscovered], color='#FF000044', marker='o', s=10)
                    axs[j,i].legend(handlelength=0.0,handletextpad=-0.1)
                    j = j + 1
                axs[0,i].set_title(key)
            i = i + 1
        axs[0,0].plot(data,linewidth=linewidth,alpha=transparency,color='black',label="ground truth")
        axs[0,0].legend()
        if multi_forecast:
            axs[1,0].plot(forecasts[:,1],linewidth=linewidth,alpha=transparency,color='green', label="forecast")
        else:
            axs[1,0].plot(forecasts,linewidth=linewidth,alpha=transparency,color='green', label="forecast")
        axs[1,0].legend()
        axs[0,0].set_title("data")
        #plt.ylim([smallest_finite_value, largest_finite_value])
        plt.ylim([data.min()*0.8, data.max()*1.2])
        fig.supxlabel('time')
        fig.supylabel(r'$\mathcal{C}_t$')
        plt.tight_layout(pad=0.05)
        plt.subplots_adjust(left=0.05, bottom=0.07)
        plt.savefig(plots_folder + "sets.pdf")

    # calculate metrics
    metrics = pd.DataFrame({})
    for key in results.keys():
        try:
            q = results[key]["q"][T_burnin+1:]
            quantile_not_infinite = np.where(~np.isinf(q))[0]
            q_clipped = np.clip(q,smallest_finite_value,largest_finite_value)
            local_metrics = {}
            local_metrics['quantile risk'] = np.maximum(alpha*(q - scores[T_burnin+1:]), (1-alpha)*(scores[T_burnin+1:] - q)).mean()
            local_metrics['quantile risk (clipped)'] = np.maximum(alpha*(q_clipped - scores[T_burnin+1:]), (1-alpha)*(scores[T_burnin+1:] - q_clipped)).mean()
            local_metrics['coverage mean'] = (q >= scores[T_burnin+1:]).astype(int).mean()
            local_metrics['coverage mean (clipped)'] = (q_clipped >= scores[T_burnin+1:]).astype(int).mean()
            local_metrics['fraction infinite'] = q[np.isinf(q)].shape[0] / q.shape[0]
            local_metrics['quantile median'] = np.median(q)
            local_metrics['largest coverage excursion'] = np.max(np.cumsum((q >= scores[T_burnin+1:]).astype(int) - np.cumsum((q < scores[T_burnin+1:]).astype(int))))
            local_metrics['largest miscoverage excursion'] = np.max(np.cumsum((q < scores[T_burnin+1:]).astype(int) - np.cumsum((q >= scores[T_burnin+1:]).astype(int))))
            local_metrics['longest coverage sequence'] = max(sum(1 for i in g) for k,g in groupby((q >= scores[T_burnin+1:]).astype(int)))
            local_metrics['longest miscoverage sequence'] = max(sum(1 for i in g) for k,g in groupby((q < scores[T_burnin+1:]).astype(int)))
            if real_data:
                sets = results[key]["sets"]
                local_metrics['median size'] = np.median(sets[1] - sets[0])
                local_metrics['25\% size'] = np.quantile(sets[1] - sets[0], 0.25)
                local_metrics['75\% size'] = np.quantile(sets[1] - sets[0], 0.75)
                local_metrics['90\% size'] = np.quantile(sets[1] - sets[0], 0.90) if local_metrics['fraction infinite'] < 0.1 else np.inf
                local_metrics['10\% size'] = np.quantile(sets[1] - sets[0], 0.10)
                local_metrics['interval score'] = ((sets[1] - sets[0]) + np.clip(np.maximum(2/alpha * (sets[0] - data), 2/alpha * (data - sets[1])), 0, None)).mean()
            lr_index = f"{lr:0.1E}" if isinstance(lr,float) else lr
            index = pd.MultiIndex.from_tuples([(key, lr_index)], names=["method", "lr"])
            local_metrics = pd.DataFrame(local_metrics, index=index)
            metrics = pd.concat([metrics, local_metrics], ignore_index=False)
        except:
            for lr in results[key].keys():
                q = results[key][lr]["q"][T_burnin+1:]
                quantile_not_infinite = np.where(~np.isinf(q))[0]
                q_clipped = np.clip(q,smallest_finite_value,largest_finite_value)
                local_metrics = {}
                local_metrics['quantile risk'] = np.maximum(alpha*(q - scores[T_burnin+1:]), (1-alpha)*(scores[T_burnin+1:] - q)).mean()
                local_metrics['quantile risk (clipped)'] = np.maximum(alpha*(q_clipped - scores[T_burnin+1:]), (1-alpha)*(scores[T_burnin+1:] - q_clipped)).mean()
                local_metrics['coverage mean'] = (q >= scores[T_burnin+1:]).astype(int).mean()
                local_metrics['coverage mean (clipped)'] = (q_clipped >= scores[T_burnin+1:]).astype(int).mean()
                local_metrics['fraction infinite'] = q[np.isinf(q)].shape[0] / q.shape[0]
                local_metrics['quantile median'] = np.median(q)
                local_metrics['largest coverage excursion'] = max(np.max(np.cumsum((q >= scores[T_burnin+1:]).astype(int) - np.cumsum((q < scores[T_burnin+1:]).astype(int)))), 0)
                local_metrics['largest miscoverage excursion'] = max(np.max(np.cumsum((q < scores[T_burnin+1:]).astype(int) - np.cumsum((q >= scores[T_burnin+1:]).astype(int)))), 0)
                local_metrics['longest coverage sequence'] = max(sum(1 for i in g) for k,g in groupby((q >= scores[T_burnin+1:]).astype(int)))
                local_metrics['longest miscoverage sequence'] = max(sum(1 for i in g) for k,g in groupby((q < scores[T_burnin+1:]).astype(int)))
                if real_data:
                    sets = results[key][lr]["sets"]
                    local_metrics['median size'] = np.median(sets[1] - sets[0])
                    local_metrics['25\% size'] = np.quantile(sets[1] - sets[0], 0.25)
                    local_metrics['75\% size'] = np.quantile(sets[1] - sets[0], 0.75)
                    local_metrics['90\% size'] = np.quantile(sets[1] - sets[0], 0.90) if local_metrics['fraction infinite'] < 0.1 else np.inf
                    local_metrics['10\% size'] = np.quantile(sets[1] - sets[0], 0.10)
                    local_metrics['interval score'] = ((sets[1] - sets[0]) + np.clip(np.maximum(2/alpha * (sets[0] - data), 2/alpha * (data - sets[1])),0,None)).mean()
                lr_index = f"{lr:0.1E}" if isinstance(lr,float) else lr
                index = pd.MultiIndex.from_tuples([(key, lr_index)], names=["method", "lr"])
                local_metrics = pd.DataFrame(local_metrics, index=index)
                metrics = pd.concat([metrics, local_metrics], ignore_index=False)
    metrics = metrics.T
    formatter = lambda x: x if isinstance(x,str) else "{:0.2E}".format(x)
    s = metrics.style.format(formatter=formatter, na_rep="")
    metrics_str = s.to_latex(sparse_index=True, sparse_columns=True, siunitx=False, hrules=True)
    # Add latex header to metrics_str
    metrics_str = r'\documentclass{article}' + "\n" + \
            r'\usepackage{booktabs}' + "\n" + \
            r'\usepackage{multirow}' + "\n" + \
            r'\usepackage{amsmath}' + "\n" + \
            r'\usepackage{amssymb}' + "\n" + \
            r'\usepackage{amsthm}' + "\n" + \
            r'\usepackage{amsfonts}' + "\n" + \
            r'\usepackage{graphicx}' + "\n" + \
            r'\usepackage{subcaption}' + "\n" + \
            r'\usepackage{caption}' + "\n" + \
            r'\usepackage{float}' + "\n" + \
            r'\usepackage{pdflscape}' + "\n" + \
            r'\usepackage{siunitx}' + "\n" + \
            r'\usepackage[paperwidth=5in, paperheight=35in]{geometry}' + "\n" + \
            r'\usepackage{longtable}' + "\n" + \
            r'\begin{document}' + "\n" + \
            r'\newgeometry{margin=1cm}' + "\n" + \
            r'\begin{landscape}' + "\n" + \
            r'\begin{table}' + "\n" + \
            metrics_str + \
            r'\end{table}' + "\n" + \
            r'\end{landscape}' + "\n" + \
            r'\end{document}'
    # Save string
    with open(metrics_folder + "metrics.tex", "w") as text_file:
        text_file.write(metrics_str)
    # Compile latex
    os.system("pdflatex -output-directory " + metrics_folder + " " + metrics_folder + "metrics.tex")
