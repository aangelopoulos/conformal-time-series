import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from core.utils import moving_average
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import pickle as pkl
from matplotlib.dates import date2num
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import pdb
import matplotlib.patheffects as pe

def plot_time_series(axs, time_series_list, window_start, window_end, sets, T_burnin, y, hline=None):
    # Create a figure and a grid of subplots
    all_axins = []
    # Get the minimum and maximum values for the axes and axins
    if not sets:
        minval_ax = min([ time_series[T_burnin:].min() for time_series in time_series_list ])
        maxval_ax = max([ time_series[T_burnin:].max() for time_series in time_series_list ])
        minval_axins = min([ time_series[window_start:window_end].min() for time_series in time_series_list ])
        maxval_axins = max([ time_series[window_start:window_end].max() for time_series in time_series_list ])
    else:
        minval_axins = min([ time_series[0][window_start:window_end].min() for time_series in time_series_list ])
        maxval_axins = max([ time_series[1][window_start:window_end].max() for time_series in time_series_list ])
        minval_ax = min([ time_series[0][T_burnin:].min() for time_series in time_series_list ])
        maxval_ax = max([ time_series[1][T_burnin:].max() for time_series in time_series_list ])
    for i, time_series in enumerate(time_series_list):
        ax = axs[i]

        # Use seaborn to plot the time series on the ax
        if not sets:
            sns.lineplot(x=time_series.index[T_burnin:], y=time_series[T_burnin:], ax=ax)
        else:
            cvds = (time_series[0] <= y) & (time_series[1] >= y)
            ax.plot(np.arange(y.shape[0])[T_burnin:], y[T_burnin:], color='black', alpha=0.3)
            idx_miscovered = np.where(1-cvds)[0]
            idx_miscovered = idx_miscovered[idx_miscovered >= T_burnin]
            ax.fill_between(np.arange(y.shape[0])[T_burnin:], np.clip(time_series[0][T_burnin:], minval_ax, maxval_ax), np.clip(time_series[1][T_burnin:], minval_ax, maxval_ax), alpha=0.3)
            #ax.scatter(idx_miscovered, y[idx_miscovered], color='#FF000044', marker='o', s=5)
        if hline is not None:
            ax.axhline(hline, color='#888888', linestyle='--', linewidth=1)
        sns.despine(ax=ax)  # Despine the top and right axes


        # Rotate x-axis labels and set their font size for better visibility
        ax.tick_params(axis='x', rotation=45)
        for label in ax.get_xticklabels():
            label.set_fontsize(12)

        # Define the inset ax in the top right corner
        axins = inset_axes(ax, width="40%", height="40%", loc='upper right', borderpad=4)

        # Give the inset a different background color
        axins.set_facecolor('whitesmoke')

        # On the inset ax, plot the same time series but only the window of interest
        if not sets:
            sns.lineplot(x=time_series[window_start:window_end].index, y=time_series[window_start:window_end], ax=axins)
        else:
            pass
            cvds = (time_series[0][window_start:window_end] <= y[window_start:window_end]) & (time_series[1][window_start:window_end] >= y[window_start:window_end])
            axins.plot(np.arange(y.shape[0])[window_start:window_end], y[window_start:window_end], color='black', alpha=0.3)
            idx_miscovered = np.where(1-cvds)[0]
            idx_miscovered = idx_miscovered[idx_miscovered >= window_start]
            axins.fill_between(np.arange(y.shape[0])[window_start:window_end], np.clip(time_series[0][window_start:window_end], minval_axins, maxval_axins), np.clip(time_series[1][window_start:window_end], minval_axins, maxval_axins), alpha=0.3)
            #axins.scatter(idx_miscovered, y[idx_miscovered], color='#FF000044', marker='o', s=5)

        if hline is not None:
            axins.axhline(hline, color='#888888', linestyle='--', linewidth=1)
        #sns.despine(ax=axins)  # Despine the top and right axes
        for axis in ['top','bottom','left','right']:
            axins.spines[axis].set_linewidth(2)
            axins.spines[axis].set_color('#FCA5AF')

        # Apply auto ticks on the inset
        axins.xaxis.set_visible(True)
        axins.yaxis.set_visible(True)

        # Rotate x-axis labels and set their font size for better visibility
        #axins.tick_params(axis='x', rotation=45)
        #for label in axins.get_xticklabels():
        #    label.set_fontsize(10)
        axins.set_xticklabels([])
        axins.set_xticks([])
        axins.set_yticklabels([])
        axins.set_yticks([])

        # Draw a box of the region of the inset axes in the parent axes and
        # connecting lines between the box and the inset axes area
        mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="#FCA5AF", lw=2)
        all_axins += [axins]

    # Set ymin and ymax for insets
    for axin in all_axins:
        axin.set_ylim(minval_axins, maxval_axins)
        # Make sure there are only two ticks on the y-axis
        #axin.yaxis.set_ticks([minval_axins, maxval_axins])

def plot_everything(coverages_list, sets_list, titles_list, y, alpha, T_burnin, window_start, window_end):
    fig, axs = plt.subplots(nrows=2, ncols=len(coverages_list), figsize=(10 * len(coverages_list), 6), sharex=True, sharey=False)
    plot_time_series(axs[0,:], coverages_list, window_start, window_end, False, T_burnin, y, hline=1-alpha)
    plot_time_series(axs[1,:], sets_list, window_start, window_end, True, T_burnin, y)
    axs[0,0].set_ylabel('coverage', fontsize=16)
    axs[1,0].set_ylabel('sets', fontsize=16)
    axs[0,0].set_title(titles_list[0], fontsize=16)
    axs[0,1].set_title(titles_list[1], fontsize=16)
    # Add a horizontal dotted line at 1-alpha for every axis
    for ax in axs[0,:]:
        ax.set_ylim([0.5,1.1])
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.subplots_adjust(left=0.05, bottom=0.15)
    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("time", fontsize=16, labelpad=20)
    plt.show()


if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Plot time series data.')
    parser.add_argument('filename', help='Path to pickle file containing time series data.')
    parser.add_argument('key1', help='First key for time series data extraction.')
    parser.add_argument('lr1', help='Learning rate associated with first key.', type=float)
    parser.add_argument('key2', help='Second key for time series data extraction.')
    parser.add_argument('lr2', help='Learning rate associated with second key.', type=float)
    parser.add_argument('window_length', help='Length of inset window.', default=60, type=int)

    # Parse command line arguments
    args = parser.parse_args()

    # Load the data from the pickle file
    with open(args.filename, 'rb') as f:
        data = pkl.load(f)

    # Extract time series data using provided keys and learning rates
    quantiles_given = data['quantiles_given']
    T_burnin = data['T_burnin']
    quantiles1 = data[args.key1][args.lr1]['q'][1:]
    quantiles2 = data[args.key2][args.lr2]['q'][1:]
    #((y[:-1] >= sets2[0]) & (y[:-1] <= sets2[1])).astype(float).mean()
    forecasts = data['forecasts'][:-1]
    alpha = data['alpha']
    scores = data['scores']
    y = data['data'][data['data']['item_id'] == 'y']['target'].to_numpy().astype(float)[:-1]
    y_clip_low = None#y.min()*0.8
    y_clip_high = np.infty#y.max()*1.2
    if quantiles_given:
        sets1 = [np.clip(forecasts[0] - quantiles1, y_clip_low, y_clip_high), np.clip(forecasts[-1] + quantiles1, y_clip_low, y_clip_high)]
        sets2 = [np.clip(forecasts[0] - quantiles2, y_clip_low, y_clip_high), np.clip(forecasts[-1] + quantiles2, y_clip_low, y_clip_high)]
    else:
        sets1 = [np.clip(forecasts - quantiles1, y_clip_low, y_clip_high), np.clip(forecasts + quantiles1, y_clip_low, y_clip_high)]
        sets2 = [np.clip(forecasts - quantiles2, y_clip_low, y_clip_high), np.clip(forecasts + quantiles2, y_clip_low, y_clip_high)]

    #covered1 = moving_average((scores <= quantiles1).astype(float))
    #covered2 = moving_average((scores <= quantiles2).astype(float))
    covered1 = moving_average(((y >= sets1[0]) & (y <= sets1[1])).astype(float))
    covered2 = moving_average(((y >= sets2[0]) & (y <= sets2[1])).astype(float))

    # Create pandas Series from the arrays with a simple numeric index
    time_series1 = pd.Series(covered1)
    time_series2 = pd.Series(covered2)

    window_end = time_series1.shape[0]
    window_start = window_end - args.window_length

    # Call the plot_time_series function to plot the data
    plot_everything([time_series1, time_series2], [sets1, sets2], [args.key1, args.key2], y, alpha, T_burnin, window_start, window_end)

