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
from matplotlib.patches import Rectangle
import pdb
import matplotlib.patheffects as pe
import matplotlib.ticker as mtick

def plot_time_series(fig, axs, time_series_list, window_start, window_end, sets, T_burnin, y, color, hline=None):
    # Create a figure and a grid of subplots
    all_axins = []
    # Get the minimum and maximum values for the axes and axins
    # Create a list of time series with only finite values. The time series are all numpy arrays
    if not sets:
        ts_list_finite = [ np.where(np.isfinite(time_series), time_series, np.nan) for time_series in time_series_list ]
    else:
        ts_list_finite = [ [np.where(np.isfinite(time_series[0]) & np.isfinite(time_series[1]), time_series[0], np.nan), np.where(np.isfinite(time_series[0]) & np.isfinite(time_series[1]), time_series[1], np.nan)] for time_series in time_series_list ]
    if not sets:
        minval_ax = min([ np.nanmin(time_series[T_burnin:]) for time_series in ts_list_finite ])
        maxval_ax = max([ np.nanmax(time_series[T_burnin:]) for time_series in ts_list_finite ])
        minval_axins = min([ np.nanmin(time_series[window_start:window_end]) for time_series in ts_list_finite ])
        maxval_axins = max([ np.nanmax(time_series[window_start:window_end]) for time_series in ts_list_finite ])
    else:
        minval_ax = min([ np.nanmin(time_series[0][T_burnin:]) for time_series in ts_list_finite ])
        maxval_ax = max([ np.nanmax(time_series[1][T_burnin:]) for time_series in ts_list_finite ])
        minval_axins = min([ np.nanmin(time_series[0][window_start:window_end]) for time_series in ts_list_finite ])
        maxval_axins = max([ np.nanmax(time_series[1][window_start:window_end]) for time_series in ts_list_finite ])

    for i, time_series in enumerate(time_series_list):
        ax = axs[i]

        # Use seaborn to plot the time series on the ax
        if not sets:
            sns.lineplot(x=time_series.index[T_burnin:], y=time_series[T_burnin:], ax=ax, color=color)
        else:
            cvds = (time_series[0] <= y) & (time_series[1] >= y)
            ax.fill_between(np.arange(y.shape[0])[T_burnin:], np.clip(time_series[0][T_burnin:], minval_ax, maxval_ax), np.clip(time_series[1][T_burnin:], minval_ax, maxval_ax), color=color)
            ax.plot(np.arange(y.shape[0])[T_burnin:], y[T_burnin:], color='black', alpha=0.3, linewidth=1)
            idx_miscovered = np.where(1-cvds)[0]
            idx_miscovered = idx_miscovered[idx_miscovered >= T_burnin]
        if hline is not None:
            ax.axhline(hline, color='#888888', linestyle='--', linewidth=1)
        sns.despine(ax=ax)  # Despine the top and right axes

        # Define the inset ax in the lower right corner
        axins = ax.inset_axes([0.6,0.05,0.4,0.4])
        #axins = inset_axes(ax, width="40%", height="40%", loc='lower center', borderpad=2)

        # Give the inset a different background color
        axins.set_facecolor('whitesmoke')

        # On the inset ax, plot the same time series but only the window of interest
        if not sets:
            sns.lineplot(x=time_series[window_start:window_end].index, y=time_series[window_start:window_end], ax=axins, color=color)
        else:
            cvds = (time_series[0][window_start:window_end] <= y[window_start:window_end]) & (time_series[1][window_start:window_end] >= y[window_start:window_end])
            axins.fill_between(np.arange(y.shape[0])[window_start:window_end], np.clip(time_series[0][window_start:window_end], minval_axins, maxval_axins), np.clip(time_series[1][window_start:window_end], minval_axins, maxval_axins), color=color)
            axins.plot(np.arange(y.shape[0])[window_start:window_end], y[window_start:window_end], color='black', alpha=0.3, linewidth=1)

        if hline is not None:
            axins.axhline(hline, color='#888888', linestyle='--', linewidth=1)

        box_color = "#dcd9d9"
        for axis in ['top','bottom','left','right']:
            axins.spines[axis].set_linewidth(2)
            axins.spines[axis].set_color(box_color)

        # Draw a box of the region of the inset axes in the parent axes and
        # connecting lines between the box and the inset axes area
        mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec=box_color, lw=2)

        # Apply auto ticks on the inset
        axins.xaxis.set_visible(True)
        axins.yaxis.set_visible(True)

        axins.set_xticklabels([])
        axins.set_xticks([])
        axins.set_yticklabels([])
        axins.set_yticks([])

        all_axins += [axins]

    # Set ymin and ymax for insets
    for axin in all_axins:
        axin.set_ylim(minval_axins-0.1*np.abs(minval_axins), maxval_axins + 0.1*np.abs(maxval_axins))

def plot_everything(coverages_list, sets_list, titles_list, y, alpha, T_burnin, window_start, window_end, savename):
    fig, axs = plt.subplots(nrows=2, ncols=len(coverages_list), figsize=(10 * len(coverages_list), 6), sharex=True, sharey=False)
    plot_time_series(fig, axs[0,:], coverages_list, window_start, window_end, False, T_burnin, y, "#138085", hline=1-alpha )
    plot_time_series(fig, axs[1,:], sets_list, window_start, window_end, True, T_burnin, y, "#EEB362")
    axs[0,0].set_ylabel('Coverage', fontsize=20)
    axs[1,0].set_ylabel('Sets', fontsize=20)
    axs[0,0].set_title(titles_list[0], fontsize=20)
    axs[0,1].set_title(titles_list[1], fontsize=20)
    # Get the max and min values of each axis in axs[1,:] by calling get_ylim
    ymin = min([ax.get_ylim()[0] for ax in axs[1,:]])
    ymax = max([ax.get_ylim()[1] for ax in axs[1,:]])
    for ax in axs[0,:]:
        ax.set_ylim([0.5,1.2])
        ax.set_yticks([0.5, 0.75, 1.0])
    axs[0,0].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    axs[0,0].yaxis.set_tick_params(labelsize=13)
    axs[0,1].set_yticklabels([])

    for ax in axs[1,:]:
        ax.set_ylim([ymin, ymax + 0.1*np.abs(ymax)])
    axs[1,1].set_yticklabels([])

    axs[1,0].yaxis.set_tick_params(labelsize=13)
    axs[0,0].yaxis.set_tick_params(labelsize=13)
    axs[1,0].xaxis.set_tick_params(labelsize=13)
    axs[1,1].xaxis.set_tick_params(labelsize=13)

    plt.subplots_adjust(left=0.1, bottom=0.15)
    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time", fontsize=20, labelpad=20)
    os.makedirs('./1v1', exist_ok=True)
    plt.savefig('./1v1/' + savename + '.pdf', bbox_inches='tight')

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Plot time series data.')
    parser.add_argument('filename', help='Path to pickle file containing time series data.')
    parser.add_argument('key1', help='First key for time series data extraction.')
    parser.add_argument('lr1', help='Learning rate associated with first key.', type=float)
    parser.add_argument('key2', help='Second key for time series data extraction.')
    parser.add_argument('lr2', help='Learning rate associated with second key.', type=float)
    parser.add_argument('window_length', help='Length of inset window.', default=60, type=int)
    parser.add_argument('window_start', help='Start of inset window.', default=None, type=int)

    method_title_map = {
        'aci': 'ACI',
        'quantile': 'Quantile tracker',
        'pi': 'Quantile tracker + Integrator',
        'pid+ets': 'Quantile tracker + Integrator + Scorecaster'
    }

    # Parse command line arguments
    args = parser.parse_args()

    args.window_start = args.window_start if args.window_start is not None else -args.window_length
    datasetname = args.filename.split('/')[-1].split('.')[0]

    # Load the data from the pickle file
    with open(args.filename, 'rb') as f:
        data = pkl.load(f)

    # Extract time series data using provided keys and learning rates
    quantiles_given = data['quantiles_given']
    T_burnin = data['T_burnin']
    quantiles1 = data[args.key1][args.lr1]['q'][1:]
    quantiles2 = data[args.key2][args.lr2]['q'][1:]
    forecasts = data['forecasts']
    # if forecasts is a list, clip off the last element of each. otherwise, clip off the last element of the array
    forecasts = [f[:-1] for f in forecasts] if isinstance(forecasts, list) else forecasts[:-1]
    alpha = data['alpha']
    scores = data['scores']
    y = data['data'][data['data']['item_id'] == 'y']['target'].to_numpy().astype(float)[:-1]
    if quantiles_given:
        sets1 = [forecasts[0] - quantiles1, forecasts[-1] + quantiles1]
        sets2 = [forecasts[0] - quantiles2, forecasts[-1] + quantiles2]
    else:
        sets1 = [forecasts - quantiles1, forecasts + quantiles1]
        sets2 = [forecasts - quantiles2, forecasts + quantiles2]

    #covered1 = moving_average((scores <= quantiles1).astype(float))
    #covered2 = moving_average((scores <= quantiles2).astype(float))
    covered1 = moving_average(((y >= sets1[0]) & (y <= sets1[1])).astype(float))
    covered2 = moving_average(((y >= sets2[0]) & (y <= sets2[1])).astype(float))

    # Create pandas Series from the arrays with a simple numeric index
    time_series1 = pd.Series(covered1)
    time_series2 = pd.Series(covered2)

    window_start = args.window_start
    window_end = args.window_start + args.window_length

    savename = datasetname + '_' + args.key1 + '_lr' + str(args.lr1) + '_' + args.key2 + '_lr' + str(args.lr2) + '_window' + str(args.window_length) + '_start' + str(args.window_start)

    # Call the plot_time_series function to plot the data
    plot_everything([time_series1, time_series2], [sets1, sets2], [method_title_map[args.key1], method_title_map[args.key2]], y, alpha, T_burnin, window_start, window_end, savename)

