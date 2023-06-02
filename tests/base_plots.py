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
    score_function_name = results["score_function_name"]
    asymmetric = results["asymmetric"]

    if real_data:
        forecasts = results["forecasts"]
        data = results["data"]
        listlike_forecast = is_listlike(forecasts[0])
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
    del results["score_function_name"]
    del results["asymmetric"]

    """

        START PLOTTING

    """
    nrows = max([len(list(results[key].keys())) for key in results.keys()])
    ncols = len(list(results.keys()))

    coverages = {}
    for key in results.keys():
        for lr in list(results[key].keys()):
            if real_data:
                results[key][lr]["covered"] = (
                          (np.stack(results[key][lr]["sets"][T_burnin+1:])[:,0] <= data['y'][T_burnin+1:]) \
                        & (np.stack(results[key][lr]["sets"][T_burnin+1:])[:,1] >= data['y'][T_burnin+1:]) \
                    )
            else:
                results[key][lr]["covered"] = results[key][lr]["q"][T_burnin+1:] >= scores[T_burnin+1:]
        coverages[key] = { lr : results[key][lr]["covered"].astype(int).mean() for lr in list(results[key].keys()) }
    print(coverages)

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
            label = f"lr={lr}, cvg={100*coverages[key][lr]:.1f}%" if lr is not None else f"cvg={100*coverages[key][lr]:.1f}%"
            axs[j,i].plot(xlabels_nonscores, moving_average(results[key][lr]["covered"]), label=label, linewidth=linewidth, color=color, alpha=transparency)
            if label is not None:
                axs[j,i].legend(handlelength=0.0,handletextpad=-0.1)
            j = j + 1
        # Split the key by the '+' character and add a new line
        title = key.replace('+', '\n+').replace('(log)', '')
        axs[0,i].set_title(title)
        i = i + 1
    fig.supxlabel('Time')
    fig.supylabel('Coverage')
    plt.tight_layout(pad=0.05)
    plt.subplots_adjust(left=0.07, bottom=0.07, right=0.95, wspace=0.2)
    plt.savefig(plots_folder + "coverage.pdf")

    # Size plots (zoomed in)! Only visualize the 'upper' score, i.e., the last one in the array.
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols+1, sharex=True, sharey=True, figsize = ((ncols + 1)*10.1, nrows*6.4))
    # Plot setup
    last = 100
    i = 1
    upper_scores = np.stack(scores)[-last:,-1] if len(np.stack(scores).shape) > 1 else scores[-last:]
    low_clip = np.nanmin(np.stack(upper_scores)[-last:]) * 0.9
    high_clip = np.nanmax(np.stack(upper_scores)[-last:]) * 1.1
    # Loop through methods
    for key in results.keys():
        color = cmap_lines[i-1]
        j = 0
        for lr in results[key].keys():
            # Get the quantiles
            upper_quantiles = np.stack(results[key][lr]["q"])[-last:,-1] if len(scores.shape) > 1 else results[key][lr]["q"][-last:]
            upper_quantiles = np.clip(upper_quantiles, low_clip, high_clip)[-last:]
            # Plot the scores and quantiles
            axs[j,i].plot(xlabels_nonscores[-last:], upper_scores,linewidth=linewidth,alpha=transparency/4,color=cmap_lines[-1])
            label = f"lr={lr}" if lr is not None else None
            axs[j,i].plot(xlabels_nonscores[-last:], upper_quantiles, linewidth=linewidth, color=color, alpha=transparency, label=label)
            if label is not None:
                axs[j,i].legend(handlelength=0.0,handletextpad=-0.1)
            j = j + 1
        title = key.replace('+', '\n+').replace('(log)', '')
        axs[0,i].set_title(title)
        i = i + 1
    axs[0,0].plot(xlabels_nonscores[-last:], upper_scores,linewidth=linewidth,alpha=transparency,color=cmap_lines[-1])
    axs[0,0].set_title("scores")
    plt.ylim([low_clip, high_clip])
    fig.supxlabel('Time')
    fig.supylabel(r'$q_t$')
    plt.tight_layout(pad=0.05)
    plt.subplots_adjust(left=0.07, bottom=0.07, right=0.95, wspace=0.2)
    plt.savefig(plots_folder + "size_zoomed.pdf")

    # Plot sets (zoomed)
    if real_data:
        sns.set_theme(context="notebook", palette=cmap_lines, style="white", font_scale=4)
        sns.set_style({'axes.spines.right': False, 'axes.spines.top': False})
        if listlike_forecast:
            forecasts_zoomed = [forecast[-last:] for forecast in forecasts]
        else:
            forecasts_zoomed = forecasts[-last:]
        y_zoomed = data['y'].to_numpy().astype(float)[-last:]
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols+1, sharex=True, sharey=True, figsize = ((ncols + 1)*10.1, nrows*6.4))
        # Find limits
        i = 1
        y_clip_low = np.nanmin(y_zoomed) * 0.8
        y_clip_high = np.nanmax(y_zoomed) * 1.2
        for key in results.keys():
            color = lighten_color(desaturate_color(cmap_lines[i-1], saturation=0.3), 0.5)
            j = 0
            for lr in results[key].keys():
                sets_zoomed = np.stack(results[key][lr]["sets"])[-last:]
                sets_zoomed = np.clip(sets_zoomed, y_clip_low, y_clip_high)
                results[key][lr]["sets_zoomed"] = sets_zoomed
                cvds_zoomed = (sets_zoomed[:,0] <= y_zoomed) & (sets_zoomed[:,1] >= y_zoomed)
                axs[j,i].plot(np.arange(y_zoomed.shape[0]), y_zoomed, color='black', alpha=0.2)
                label = f"lr={lr}" if lr is not None else None
                axs[j,i].fill_between(np.arange(y_zoomed.shape[0]), sets_zoomed[:,0], sets_zoomed[:,1], color=color, alpha=transparency, label=label)
                if label is not None:
                    axs[j,i].legend(handlelength=0.0,handletextpad=-0.1)
                j = j + 1
            title = key.replace('+', '\n+')
            axs[0,i].set_title(title)
            i = i + 1
        axs[0,0].plot(np.arange(y_zoomed.shape[0]),y_zoomed,linewidth=linewidth,alpha=transparency,color='black',label="ground truth")
        axs[0,0].legend()
        if listlike_forecast:
            axs[1,0].plot(np.array(forecasts_zoomed[1]),linewidth=linewidth,alpha=transparency,color='green', label="forecast")
        else:
            axs[1,0].plot(forecasts_zoomed.to_numpy(),linewidth=linewidth,alpha=transparency,color='green', label="forecast")
        axs[1,0].legend()
        axs[0,0].set_title("y")
        plt.ylim([y_clip_low, y_clip_high])
        fig.supxlabel('Time')
        fig.supylabel(r'$\mathcal{C}_t$')
        plt.tight_layout(pad=0.05)
        plt.subplots_adjust(left=0.07, bottom=0.07, right=0.95, wspace=0.2)
        plt.savefig(plots_folder + "sets_zoomed.pdf")

    # Plot sets
    if real_data:
        sns.set_theme(context="notebook", palette=cmap_lines, style="white", font_scale=4)
        sns.set_style({'axes.spines.right': False, 'axes.spines.top': False})
        if listlike_forecast:
            forecasts_zoomed = [forecast[T_burnin+1:] for forecast in forecasts]
        else:
            forecasts_zoomed = forecasts[T_burnin+1:]
        y_zoomed = data['y'].to_numpy().astype(float)[T_burnin+1:]
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols+1, sharex=True, sharey=True, figsize = ((ncols + 1)*10.1, nrows*6.4))
        # Find limits
        i = 1
        y_clip_low = np.nanmin(y_zoomed) * 0.8
        y_clip_high = np.nanmax(y_zoomed) * 1.2
        for key in results.keys():
            color = lighten_color(desaturate_color(cmap_lines[i-1], saturation=0.3), 0.5)
            j = 0
            for lr in results[key].keys():
                sets_zoomed = np.stack(results[key][lr]["sets"])[T_burnin+1:]
                sets_zoomed = np.clip(sets_zoomed, y_clip_low, y_clip_high)
                results[key][lr]["sets_zoomed"] = sets_zoomed
                cvds_zoomed = (sets_zoomed[:,0] <= y_zoomed) & (sets_zoomed[:,1] >= y_zoomed)
                axs[j,i].plot(np.arange(y_zoomed.shape[0]), y_zoomed, color='black', alpha=0.2)
                label = f"lr={lr}" if lr is not None else None
                axs[j,i].fill_between(np.arange(y_zoomed.shape[0]), sets_zoomed[:,0], sets_zoomed[:,1], color=color, alpha=transparency, label=label)
                if label is not None:
                    axs[j,i].legend(handlelength=0.0,handletextpad=-0.1)
                j = j + 1
            title = key.replace('+', '\n+')
            axs[0,i].set_title(title)
            i = i + 1
        axs[0,0].plot(y_zoomed,linewidth=linewidth,alpha=transparency,color='black',label="ground truth")
        axs[0,0].legend()
        if listlike_forecast:
            axs[1,0].plot(np.array(forecasts_zoomed[1]),linewidth=linewidth,alpha=transparency,color='green', label="forecast")
        else:
            axs[1,0].plot(forecasts_zoomed.to_numpy(),linewidth=linewidth,alpha=transparency,color='green', label="forecast")
        axs[1,0].legend()
        axs[0,0].set_title("y")
        plt.ylim([y_clip_low, y_clip_high])
        fig.supxlabel('Time')
        fig.supylabel(r'$\mathcal{C}_t$')
        plt.tight_layout(pad=0.05)
        plt.subplots_adjust(left=0.07, bottom=0.07, right=0.95, wspace=0.2)
        plt.savefig(plots_folder + "sets.pdf")
