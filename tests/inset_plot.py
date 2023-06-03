import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from plotting_utils import moving_average, plot_time_series
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

def plot_everything(coverages_list, sets_list, titles_list, y, alpha, T_burnin, window_start, window_end, window_loc, coverage_inset, set_inset, savename):
    fig, axs = plt.subplots(nrows=2, ncols=len(coverages_list), figsize=(10 * len(coverages_list), 6), sharex=True, sharey=False)
    plot_time_series(fig, axs[0,:], coverages_list, window_start, window_end, window_loc, False, T_burnin, y, "#138085", coverage_inset, hline=1-alpha )
    plot_time_series(fig, axs[1,:], sets_list, window_start, window_end, window_loc, True, T_burnin, y, "#EEB362", set_inset)
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
    os.makedirs('./plots/', exist_ok=True)
    os.makedirs('./plots/1v1', exist_ok=True)
    plt.savefig('./plots/1v1/' + savename + '.pdf', bbox_inches='tight')

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Plot time series data.')
    parser.add_argument('--filename', help='Path to pickle file containing time series data.')
    parser.add_argument('--key1', help='First key for time series data extraction.')
    parser.add_argument('--lr1', help='Learning rate associated with first key.', type=float)
    parser.add_argument('--key2', help='Second key for time series data extraction.')
    parser.add_argument('--lr2', help='Learning rate associated with second key.', type=float)
    parser.add_argument('--window_length', help='Length of inset window.', default=60, type=int)
    parser.add_argument('--window_start', help='Start of inset window.', default=None, type=int)
    parser.add_argument('--window_loc', help='Location of inset window.', default='upper right', type=str)
    parser.add_argument('--coverage_average_length', help='Length of moving average window for coverage.', default=50, type=int)
    parser.add_argument('--coverage_inset', dest='coverage_inset', default=False, action='store_true')
    parser.add_argument('--set_inset', dest='set_inset', default=False, action='store_true')

    method_title_map = {
        'ACI': 'ACI',
        'Quantile': 'Quantile tracker',
        'Quantile+Integrator (log)': 'Quantile tracker + Integrator',
        'Quantile+Integrator (log)+Scorecaster': 'Quantile tracker + Integrator + Scorecaster'
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
    sets1 = [np.stack(data[args.key1][args.lr1]['sets'])[T_burnin+1:,0], np.stack(data[args.key1][args.lr1]['sets'])[T_burnin+1:,1]]
    sets2 = [np.stack(data[args.key2][args.lr2]['sets'])[T_burnin+1:,0], np.stack(data[args.key2][args.lr2]['sets'])[T_burnin+1:,1]]
    forecasts = data['forecasts']
    asymmetric = data['asymmetric']
    # if forecasts is a list, clip off the last element of each. otherwise, clip off the last element of the array
    forecasts = [f[T_burnin+1:] for f in forecasts] if isinstance(forecasts, list) else forecasts[:-1]
    alpha = data['alpha']
    scores = data['scores'][T_burnin+1]
    y = data['data']['y'].to_numpy().astype(float)[T_burnin+1:]

    covered1 = moving_average(((y >= sets1[0]) & (y <= sets1[1])).astype(float), args.coverage_average_length)
    covered2 = moving_average(((y >= sets2[0]) & (y <= sets2[1])).astype(float), args.coverage_average_length)

    # Create pandas Series from the arrays with a simple numeric index
    time_series1 = pd.Series(covered1)
    time_series2 = pd.Series(covered2)

    window_start = args.window_start
    window_end = args.window_start + args.window_length

    savename = datasetname + '_' + args.key1 + '_lr' + str(args.lr1) + '_' + args.key2 + '_lr' + str(args.lr2) + '_window' + str(args.window_length) + '_start' + str(args.window_start) + str(args.coverage_inset) + str(args.set_inset)

    # Call the plot_time_series function to plot the data
    plot_everything([time_series1, time_series2], [sets1, sets2], [method_title_map[args.key1], method_title_map[args.key2]], y, alpha, T_burnin, window_start, window_end, args.window_loc, args.coverage_inset, args.set_inset, savename)
