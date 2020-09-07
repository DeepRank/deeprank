#!/usr/bin/env python
"""
Plot training and validation accuracy for each epoch

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

def plt_acc(h5, figname):
    """Plot the acc vs the epoch.

    Args:
        figname (str): name of the file where to export the figure
    """

    color_plot = ['red', 'blue']
    labels = ['Train', 'Valid']
    acc = ['train', 'valid']

    fig, ax = plt.subplots()
    for ik, name in enumerate(acc):
        plt.plot(range(1,31,1),h5['acc'][name],
                    c = color_plot[ik],
                    label = labels[ik])


    plt.ylabel('Success Rate on Top N Models', fontsize='large')
    legend = ax.legend(loc='upper left')
    ax.set_xlabel('Epoch', fontsize='large', fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize='large', fontweight='bold')
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    fig.set_size_inches(5,5)
    fig.savefig('acc_'+figname)
    plt.close()


if __name__ == "__main__":

    check_input(sys.argv[1:])

    h5 = h5py.File(sys.argv[1], 'r')
    figname = sys.argv[2]
    plt_acc(h5, figname)
