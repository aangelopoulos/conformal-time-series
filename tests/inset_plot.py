import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def plot_time_series(time_series_list, window_start, window_end):
    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(nrows=1, ncols=len(time_series_list), figsize=(10 * len(time_series_list), 6), constrained_layout=True)

    for i, time_series in enumerate(time_series_list):
        ax = axs[i]

        # Use seaborn to plot the time series on the ax
        sns.lineplot(x=time_series.index, y=time_series, ax=ax)
        sns.despine(ax=ax)  # Despine the top and right axes

        # Rotate x-axis labels and set their font size for better visibility
        ax.tick_params(axis='x', rotation=45)
        for label in ax.get_xticklabels():
            label.set_fontsize(12)

        # Define the inset ax in the top right corner
        axins = inset_axes(ax, width="40%", height="30%", loc='upper right', borderpad=4)

        # Give the inset a different background color
        axins.set_facecolor('whitesmoke')

        # On the inset ax, plot the same time series but only the window of interest
        sns.lineplot(x=time_series[window_start:window_end].index, y=time_series[window_start:window_end], ax=axins)
        sns.despine(ax=axins)  # Despine the top and right axes

        # Apply auto ticks on the inset
        axins.xaxis.set_visible(True)
        axins.yaxis.set_visible(True)

        # Rotate x-axis labels and set their font size for better visibility
        axins.tick_params(axis='x', rotation=45)
        for label in axins.get_xticklabels():
            label.set_fontsize(12)

        # Draw a box of the region of the inset axes in the parent axes and
        # connecting lines between the box and the inset axes area
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.show()

# Example usage
np.random.seed(0)
dates = pd.date_range(start='1/1/2020', end='1/1/2023')
data1 = np.random.randn(len(dates))
time_series1 = pd.Series(data1, index=dates)

data2 = np.random.randn(len(dates))
time_series2 = pd.Series(data2, index=dates)

window_start = '2022-01-01'
window_end = '2022-12-31'

plot_time_series([time_series1, time_series2], window_start, window_end)
