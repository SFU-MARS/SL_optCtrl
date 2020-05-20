import os, errno

import matplotlib
matplotlib.use('Agg');

from matplotlib import rc
rc('font',**{'family':'serif'})

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
DPI = 100


########
# Plot training or evaluating result
########
def plot_performance(x, y,
                     title=None,
                     xlabel=None,
                     ylabel=None,
                     figfile=None,
                     pickle=False):
    print('plot_performance', flush=True);
    # plt.rcParams["axes.edgecolor"] = "0.15"
    # plt.rcParams["axes.grid"] = True
    fig, ax = plt.subplots()

    ax.plot(x, y)

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)


    # NOTE: new code for saving performance data
    if pickle:
        pkl.dump(fig,open(figfile,'wb'))

    if figfile is None:
        plt.show()
    else:
        fig.savefig(figfile + '.pdf', dpi=DPI, transparent=True)
        plt.close(fig)
    print('plot_performance end', flush=True)

#################
# OS Operations #
#################
def maybe_mkdir(dirname):
    # Thread-safe way to create a directory if one doesn't exist
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    print('Successfully created', dirname, flush=True)


########
# distance #
########
def Euclid_dis(p1,p2):
    return np.sqrt(np.power(p1[0] - p2[0],2) + np.power(p1[1] - p2[1],2))

