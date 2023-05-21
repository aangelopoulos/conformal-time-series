import os, sys, inspect
from itertools import groupby
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle as pkl
import seaborn as sns
import pdb
from plotting_utils import *

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

    # Process results
    alpha = results["alpha"]
    scores = results["scores"]
    T_burnin = results["T_burnin"]
    real_data = results["real_data"]
    multiple_series = results["multiple_series"]
    quantiles_given = results["quantiles_given"]
    asymmetric = results["asymmetric"]
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
    del results["asymmetric"]

    """

        START PLOTTING

    """
    nrows = max([len(list(results[key].keys())) for key in results.keys()])
    ncols = len(list(results.keys()))

    coverages = {}
    for key in results.keys():
        coverages[key] = { lr : (results[key][lr]["q"][T_burnin+1:] >= scores[T_burnin+1:]).astype(int).mean() for lr in list(results[key].keys()) }
    print(coverages)

    # For diagnostic purposes, store the indexes of the miscoverages
    miscoverage_indexes = {}
    for key in results.keys():
        miscoverage_indexes[key] = { lr : np.where(results[key][lr]["q"][T_burnin+1:] < scores[T_burnin+1:])[0] for lr in list(results[key].keys()) }

    # Plot coverage
    linewidth = 2
    transparency = 0.7

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize = (ncols*10.1, nrows*6.4))
    i = 0
    xlabels_nonscores = range(T_burnin+1, scores.shape[0])
    for key in results.keys():
        color = cmap_lines[i]
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
    # Make plots
    i = 1
    low_clip = scores.min() * 0.9
    high_clip = scores.max() * 1.1
    for key in results.keys():
        color = cmap_lines[i-1]
        j = 0
        for lr in results[key].keys():
            quantiles = np.clip(results[key][lr]["q"][T_burnin+1:], low_clip, high_clip)
            axs[j,i].plot(xlabels_nonscores, scores[T_burnin+1:],linewidth=linewidth,alpha=transparency/4,color=cmap_lines[-1])
            axs[j,i].plot(xlabels_nonscores, quantiles, linewidth=linewidth, color=color, alpha=transparency, label=f"lr={lr}")
            axs[j,i].legend(handlelength=0.0,handletextpad=-0.1)
            j = j + 1
        axs[0,i].set_title(key)
        i = i + 1
    axs[0,0].plot(scores,linewidth=linewidth,alpha=transparency,color=cmap_lines[-1])
    axs[0,0].set_title("scores")
    plt.ylim([low_clip, high_clip])
    fig.supxlabel('time')
    fig.supylabel(r'$\hat{q}$')
    plt.tight_layout(pad=0.05)
    plt.subplots_adjust(left=0.05, bottom=0.07)
    plt.savefig(plots_folder + "size.pdf")

    # Size plots (zoomed in)!
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols+1, sharex=True, sharey=True, figsize = ((ncols + 1)*10.1, nrows*6.4))
    # Make plots
    last = 80
    i = 1
    low_clip = scores[-last:].min() * 0.9
    high_clip = scores[-last:].max() * 1.1
    for key in results.keys():
        color = cmap_lines[i-1]
        j = 0
        for lr in results[key].keys():
            quantiles = np.clip(results[key][lr]["q"][-last:], low_clip, high_clip)
            axs[j,i].plot(xlabels_nonscores[-last:], scores[-last:],linewidth=linewidth,alpha=transparency/4,color=cmap_lines[-1])
            axs[j,i].plot(xlabels_nonscores[-last:], quantiles, linewidth=linewidth, color=color, alpha=transparency, label=f"lr={lr}")
            axs[j,i].legend(handlelength=0.0,handletextpad=-0.1)
            j = j + 1
        axs[0,i].set_title(key)
        i = i + 1
    axs[0,0].plot(xlabels_nonscores[-last:], scores[-last:],linewidth=linewidth,alpha=transparency,color=cmap_lines[-1])
    axs[0,0].set_title("scores")
    plt.ylim([low_clip, high_clip])
    fig.supxlabel('time')
    fig.supylabel(r'$\hat{q}$')
    plt.tight_layout(pad=0.05)
    plt.subplots_adjust(left=0.05, bottom=0.07)
    plt.savefig(plots_folder + "size_zoomed.pdf")

    # Size-score-corr!
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize = (ncols*10.1, nrows*6.4))
    # Make plots
    low_clip = scores.min() * 0.9
    high_clip = scores.max() * 1.1
    i = 0
    for key in results.keys():
        color = cmap_lines[i]
        j = 0
        for lr in results[key].keys():
            quantiles = np.clip(results[key][lr]["q"][T_burnin+1:], low_clip, high_clip)
            sns.kdeplot(ax=axs[j,i],x=scores[T_burnin+1:], y=quantiles, color=color, fill=True, hue_norm=(0,1), alpha=transparency, label=f"lr={lr}")
            axs[j,i].legend(handlelength=0.0,handletextpad=-0.1)
            j = j + 1
        axs[0,i].set_title(key)
        i = i + 1
    plt.ylim([low_clip, high_clip])
    fig.supxlabel('score')
    fig.supylabel(r'$\hat{q}$')
    plt.tight_layout(pad=0.05)
    plt.subplots_adjust(left=0.05, bottom=0.07)
    plt.savefig(plots_folder + "corr.pdf")

    # Plot sets (zoomed)
    if real_data:
        sns.set_theme(context="notebook", palette=cmap_lines, style="white", font_scale=4)
        sns.set_style({'axes.spines.right': False, 'axes.spines.top': False})
        if quantiles_given:
            forecasts_zoomed = [forecast[-last:] for forecast in forecasts]
        else:
            forecasts_zoomed = forecasts[-last:]
        y_zoomed = data[data['item_id'] == 'y']['target'].to_numpy().astype(float)[-last:]
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols+1, sharex=True, sharey=True, figsize = ((ncols + 1)*10.1, nrows*6.4))
        # Find limits
        i = 1
        y_clip_low = y_zoomed.min() * 0.8
        y_clip_high = y_zoomed.max() * 1.2
        for key in results.keys():
            color = lighten_color(desaturate_color(cmap_lines[i-1], saturation=0.3), 0.5)
            j = 0
            for lr in results[key].keys():
                quantiles_zoomed = results[key][lr]["q"][-last:]
                if quantiles_given:
                    sets_zoomed = [np.clip(forecasts_zoomed[0] - quantiles_zoomed, y_clip_low, y_clip_high), np.clip(forecasts_zoomed[-1] + quantiles_zoomed, y_clip_low, y_clip_high)]
                else:
                    sets_zoomed = [np.clip(forecasts_zoomed - quantiles_zoomed, y_clip_low, y_clip_high), np.clip(forecasts_zoomed + quantiles_zoomed, y_clip_low, y_clip_high)]
                results[key][lr]["sets_zoomed"] = sets_zoomed
                cvds_zoomed = (sets_zoomed[0] <= y_zoomed) & (sets_zoomed[1] >= y_zoomed)
                axs[j,i].plot(np.arange(y_zoomed.shape[0]), y_zoomed, color='black', alpha=0.2)
                idx_miscovered = np.where(1-cvds_zoomed)[0]
                axs[j,i].fill_between(np.arange(y_zoomed.shape[0]), sets_zoomed[0], sets_zoomed[1], color=color, alpha=transparency, label=f"lr={lr}")
                axs[j,i].scatter(idx_miscovered, y_zoomed[idx_miscovered], color='#FF000044', marker='o', s=10)
                axs[j,i].legend(handlelength=0.0,handletextpad=-0.1)
                j = j + 1
            axs[0,i].set_title(key)
            i = i + 1
        axs[0,0].plot(y_zoomed,linewidth=linewidth,alpha=transparency,color='black',label="ground truth")
        axs[0,0].legend()
        if quantiles_given:
            axs[1,0].plot(forecasts_zoomed[1],linewidth=linewidth,alpha=transparency,color='green', label="forecast")
        else:
            axs[1,0].plot(forecasts_zoomed,linewidth=linewidth,alpha=transparency,color='green', label="forecast")
        axs[1,0].legend()
        axs[0,0].set_title("y")
        plt.ylim([y_clip_low, y_clip_high])
        fig.supxlabel('time')
        fig.supylabel(r'$\mathcal{C}_t$')
        plt.tight_layout(pad=0.05)
        plt.subplots_adjust(left=0.05, bottom=0.07)
        plt.savefig(plots_folder + "sets_zoomed.pdf")

    # Plot sets
    if real_data:
        sns.set_theme(context="notebook", palette=cmap_lines, style="white", font_scale=4)
        sns.set_style({'axes.spines.right': False, 'axes.spines.top': False})
        if quantiles_given:
            forecasts = [forecast[T_burnin+1:] for forecast in forecasts]
        else:
            forecasts = forecasts[T_burnin+1:]
        y = data[data['item_id'] == 'y']['target'].to_numpy().astype(float)[T_burnin+1:]
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols+1, sharex=True, sharey=True, figsize = ((ncols + 1)*10.1, nrows*6.4))
        # Find limits
        i = 1
        y_clip_low = y.min() * 0.8
        y_clip_high = y.max() * 1.2
        for key in results.keys():
            color = lighten_color(desaturate_color(cmap_lines[i-1], saturation=0.3), 0.5)
            j = 0
            for lr in results[key].keys():
                quantiles = results[key][lr]["q"][T_burnin+1:]
                if quantiles_given:
                    sets = [np.clip(forecasts[0] - quantiles, y_clip_low, y_clip_high), np.clip(forecasts[-1] + quantiles, y_clip_low, y_clip_high)]
                else:
                    sets = [np.clip(forecasts - quantiles, y_clip_low, y_clip_high), np.clip(forecasts + quantiles, y_clip_low, y_clip_high)]
                results[key][lr]["sets"] = sets
                cvds = (sets[0] <= y) & (sets[1] >= y)
                axs[j,i].plot(np.arange(y.shape[0]), y, color='black', alpha=0.2)
                idx_miscovered = np.where(1-cvds)[0]
                axs[j,i].fill_between(np.arange(y.shape[0]), sets[0], sets[1], color=color, alpha=transparency, label=f"lr={lr}")
                axs[j,i].scatter(idx_miscovered, y[idx_miscovered], color='#FF000044', marker='o', s=10)
                axs[j,i].legend(handlelength=0.0,handletextpad=-0.1)
                j = j + 1
            axs[0,i].set_title(key)
            i = i + 1
        axs[0,0].plot(y,linewidth=linewidth,alpha=transparency,color='black',label="ground truth")
        axs[0,0].legend()
        if quantiles_given:
            axs[1,0].plot(forecasts[1],linewidth=linewidth,alpha=transparency,color='green', label="forecast")
        else:
            axs[1,0].plot(forecasts,linewidth=linewidth,alpha=transparency,color='green', label="forecast")
        axs[1,0].legend()
        axs[0,0].set_title("y")
        plt.ylim([y_clip_low, y_clip_high])
        fig.supxlabel('time')
        fig.supylabel(r'$\mathcal{C}_t$')
        plt.tight_layout(pad=0.05)
        plt.subplots_adjust(left=0.05, bottom=0.07)
        plt.savefig(plots_folder + "sets.pdf")

    # calculate metrics
    metrics = pd.DataFrame({})
    for key in results.keys():
        for lr in results[key].keys():
            q = results[key][lr]["q"][T_burnin+1:]
            quantile_not_infinite = np.where(~np.isinf(q))[0]
            q_clipped = np.clip(q,scores.min(),scores.max())
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
                local_metrics['interval score'] = ((sets[1] - sets[0]) + np.clip(np.maximum(2/alpha * (sets[0] - y), 2/alpha * (y - sets[1])),0,None)).mean()
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
    os.system("pdflatex -output-directory " + metrics_folder + " " + metrics_folder + "metrics.tex > /dev/null")
