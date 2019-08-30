import matplotlib
matplotlib.use('Agg');

from matplotlib import rc
rc('font',**{'family':'serif'})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
DPI = 100

def plot_performance(x, y, 
                     title=None, 
                     xlabel=None, 
                     ylabel=None,
                     figfile=None,
                     pickle = False):
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