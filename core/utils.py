import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def moving_average(x, window=50):
    norm_factor = window / np.convolve( np.ones_like(x), np.ones(window), 'same' ) # Deal with edge effects
    return norm_factor * (np.convolve(x, np.ones(window), 'same') / window)

def plot_sets(ax,args):
    gt, sets = args['gt'], args['sets']
    cvds = (sets[0] <= gt) & (sets[1] >= gt)
    ax.plot(np.arange(gt.shape[0]), gt, color='#00FF0033', label='ground truth')
    idx_miscovered = np.where(1-cvds)[0]
    ax.fill_between(np.arange(gt.shape[0]), sets[0], sets[1], color='#0000FF33', label="sets");
    ax.scatter(idx_miscovered, gt[idx_miscovered], color='#FF000066', label='miscovered', s=2)
    ax.locator_params(tight=True, nbins=4)
    ax.set_ylabel(r'Y')
    ax.legend(loc='center left')
    sns.despine(ax=ax,top=True,right=True)

def plot_coverage_size(results, labels, colors=None, changepoint=None):
    # Plot prediction sets and coverage over time
    fig, axs = plt.subplots(2,1,figsize=(6.4,7))

    coverages = [moving_average(x['covereds']) for x in results]
    sizes = [x['sets'][1]-x['sets'][0] for x in results]

    cvg_ylims = [min([x.min() for x in coverages]), max([x.max() for x in coverages])]
    # clip inf sizes
    maxval = max([x[x != np.inf].max() for x in sizes])
    for x in sizes:
        x[x == np.inf] = maxval
    size_ylims = [min([x.min() for x in sizes]), max([np.quantile(x,0.98) for x in sizes])]
    for i in range(len(labels)):
        axs[0].plot(coverages[i], label=labels[i], color=colors[i])
    if not (changepoint is None):
        axs[0].axvline(x=changepoint, label=r'$\Delta$ pt', color='gray', linestyle='dotted')
    sns.despine(ax=axs[0],top=True,right=True)
    axs[0].set_xlabel('')
    axs[0].set_ylabel(f'coverage\n(size 50 sliding window)')
    axs[0].set_ylim(cvg_ylims)
    axs[0].locator_params(tight=True, nbins=4)

    for i in range(len(labels)):
        axs[1].plot(sizes[i], label=labels[i], color=colors[i])
    if not (changepoint is None):
        axs[1].axvline(x=changepoint, label=r'$\Delta$ pt', color='gray', linestyle='dotted')
    axs[1].set_ylim(size_ylims)
    sns.despine(ax=axs[0],top=True,right=True)
    axs[1].set_xlabel('timestamp')
    axs[1].set_ylabel(f'size')
    axs[1].locator_params(tight=True, nbins=4)
    axs[1].legend(loc='best')
    sns.despine(ax=axs[1],top=True,right=True)
    plt.tight_layout()
    plt.show()

def plot_vertical_spread(plot_args):
    plot_args["cvg"] = moving_average(plot_args["covereds"])
    plot_args["size"] = plot_args["sets"][1]-plot_args["sets"][0]

    plotkeys =  list(set(plot_args.keys()) - set(["covereds", "sets", "color", "gt", "changepoint"])) # Everything is a plot except for the color, ground truth, and changepoint

    num_plots = len(plotkeys)+1
    fig, axs = plt.subplots(nrows=num_plots, ncols=1, sharex=True, figsize=(6.4,9))

    plot_sets(axs[0],plot_args)

    for j in range(1,len(plotkeys)+1):
        axs[j].plot(plot_args[plotkeys[j-1]], color=plot_args["color"])
        try:
            axs[j].axvline(x=plot_args["changepoint"], label=r'$\Delta$ pt', color='gray', linestyle='dotted')
        except:
            print("No changepoint")
        axs[j].set_ylabel(plotkeys[j-1])
        sns.despine(ax=axs[j], top=True, right=True)
    plt.tight_layout()
    plt.show()
