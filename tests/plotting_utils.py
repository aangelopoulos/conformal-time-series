import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from itertools import groupby
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Rectangle
from matplotlib.dates import date2num
import matplotlib.patheffects as pe
import matplotlib.ticker as mtick
import pickle as pkl
import seaborn as sns
import pdb

def longest_true_sequence(arr):
    # Add False at each end to make sure we always get the length of every True sequence
    # Even when they are at the start or end of the array
    arr = np.concatenate(([False], arr, [False]))
    # Find the indices where the array goes from False to True or True to False
    idx = np.flatnonzero(arr[1:] != arr[:-1])
    # Get lengths of True sequences and get the maximum length
    max_length = np.max(idx[1::2] - idx[::2])
    return max_length

def is_listlike(l):
    return isinstance(l, (pd.Series, list, np.ndarray))

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

def moving_average(x, window=50):
    x = np.array(x)
    ma = np.zeros(x.shape)
    for i in range(x.shape[0]):
        ma[i] = np.mean(x[max(0, i-window+1):i+1])
    return ma

def plot_time_series(fig, axs, time_series_list, window_start, window_end, window_loc, sets, y, color, inset, miscoverage_scatterplot, datetimes, hline=None):
    # Create a figure and a grid of subplots
    all_axins = []
    # Get the minimum and maximum values for the axes and axins
    # Create a list of time series with only finite values. The time series are all numpy arrays
    if not sets:
        ts_list_finite = [ np.where(np.isfinite(time_series), time_series, np.nan) for time_series in time_series_list ]
    else:
        ts_list_finite = [ [np.where(np.isfinite(time_series[0]) & np.isfinite(time_series[1]), time_series[0], np.nan), np.where(np.isfinite(time_series[0]) & np.isfinite(time_series[1]), time_series[1], np.nan)] for time_series in time_series_list ]
    if not sets:
        minval_ax = np.nanmin([ np.nanmin(time_series) for time_series in ts_list_finite ])
        maxval_ax = np.nanmax([ np.nanmax(time_series) for time_series in ts_list_finite ])
        minval_axins = np.nanmin([ np.nanmin(time_series[window_start:window_end]) for time_series in ts_list_finite ])
        maxval_axins = np.nanmax([ np.nanmax(time_series[window_start:window_end]) for time_series in ts_list_finite ])
    else:
        minval_ax = np.nanmin([ np.nanmin(time_series[0]) for time_series in ts_list_finite ])
        maxval_ax = np.nanmax([ np.nanmax(time_series[1]) for time_series in ts_list_finite ])
        minval_axins = np.nanmin([ np.nanmin(time_series[0][window_start:window_end]) for time_series in ts_list_finite ])
        maxval_axins = np.nanmax([ np.nanmax(time_series[1][window_start:window_end]) for time_series in ts_list_finite ])

    for i, time_series in enumerate(time_series_list):
        ax = axs[i]

        # Use seaborn to plot the time series on the ax
        if not sets:
            sns.lineplot(x=datetimes, y=time_series, ax=ax, color=color)
        else:
            cvds = (time_series[0] <= y) & (time_series[1] >= y)
            ax.fill_between(datetimes, np.clip(time_series[0], minval_ax, maxval_ax), np.clip(time_series[1], minval_ax, maxval_ax), color=color)
            ax.plot(datetimes, y, color='black', alpha=0.3, linewidth=1)
            if miscoverage_scatterplot:
                ax.scatter(datetimes[~cvds], y[~cvds], color='red', alpha=0.7, linewidth=1, s=30)
        if hline is not None:
            ax.axhline(hline, color='#888888', linestyle='--', linewidth=1)
        sns.despine(ax=ax)  # Despine the top and right axes

        if inset:
            # Define the inset ax in the lower right corner
            if window_loc == 'lower right':
                axins = ax.inset_axes([0.6,0.05,0.4,0.4])
            elif window_loc == 'upper right':
                axins = ax.inset_axes([0.6,0.6,0.4,0.4])
            elif window_loc == 'upper left':
                axins = ax.inset_axes([0.05,0.6,0.4,0.4])

            # Give the inset a different background color
            axins.set_facecolor('whitesmoke')

            # On the inset ax, plot the same time series but only the window of interest
            if not sets:
                sns.lineplot(x=datetimes[window_start:window_end], y=time_series[window_start:window_end], ax=axins, color=color)
            else:
                cvds = (time_series[0][window_start:window_end] <= y[window_start:window_end]) & (time_series[1][window_start:window_end] >= y[window_start:window_end])
                axins.fill_between(datetimes[window_start:window_end], np.clip(time_series[0][window_start:window_end], minval_axins, maxval_axins), np.clip(time_series[1][window_start:window_end], minval_axins, maxval_axins), color=color)
                axins.plot(datetimes[window_start:window_end], y[window_start:window_end], color='black', alpha=0.3, linewidth=1)

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
