#!/usr/bin/env python
"""
Plot training and validation losses for each epoch

Usage: python {0} <deeprank training results in hdf5 format>  <output figure name>
Example: python {0} epoch_data.hdf5 fig_001

Author: {1} ({2})
"""
import sys
import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

__author__ = "Cunliang Geng"
__email__ = "gengcunliang AT gmail.com"
USAGE = __doc__.format(__file__, __author__, __email__)

def check_input(args):
    if len(args) != 2:
        sys.stderr.write(USAGE)
        sys.exit(1)

def plt_loss(h5, figname):
    """Plot the losses vs the epoch.

    Args:
        figname (str): name of the file where to export the figure
    """

    color_plot = ['red', 'blue']
    labels = ['Train', 'Valid']
    losses = ['train', 'valid']

    fig, ax = plt.subplots()
    for ik, name in enumerate(losses):
        plt.plot(range(1,31,1),h5['losses'][name],
                    c = color_plot[ik],
                    label = labels[ik])


    plt.ylabel('Success Rate on Top N Models', fontsize='large')
    legend = ax.legend(loc='upper left')
    ax.set_xlabel('Epoch', fontsize='large', fontweight='bold')
    ax.set_ylabel('Loss', fontsize='large', fontweight='bold')
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    fig.set_size_inches(5,5)
    fig.savefig(f'loss_'+figname)
    plt.close()


if __name__ == "__main__":

    check_input(sys.argv[1:])

    h5 = h5py.File(sys.argv[1], 'r')
    figname = sys.argv[2]
    plt_loss(h5, figname)
