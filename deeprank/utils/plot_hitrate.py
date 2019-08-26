import os
import sys

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from deeprank.learn import rankingMetrics
from ggplot import *


def plot_boxplot_todo(hdf5, epoch=None, figname=None, inverse=False):
    '''
    Plot a boxplot of predictions VS targets useful '
    to visualize the performance of the training algorithm
    This is only usefull in classification tasks

    Args:
        figname (str): filename

    '''

    print('\n --> Box Plot : ', figname, '\n')

    color_plot = {'train': 'red', 'valid': 'blue', 'test': 'green'}
    labels = ['train', 'valid', 'test']

    # -- read data
    h5 = h5py.File(hdf5, 'r')
    if epoch is None:
        keys = list(h5.keys())
        last_epoch_key = list(filter(lambda x: 'epoch_' in x, keys))[-1]
    else:
        last_epoch_key = 'epoch_%04d' % epoch
        if last_epoch_key not in h5:
            print(
                'Incorrect epcoh name\n Possible options are: ' +
                ' '.join(
                    list(
                        h5.keys())))
            h5.close()
            return
    h5data = h5[last_epoch_key]

    print(f"Generate boxplot for {last_epoch_key} epoch ...")

    n_panels = len(labels)
    data = pd.DataFrame()

    for l in labels:

        if l in h5data:

            tar = h5data[l]['targets']
            raw_out = h5data[l]['outputs']

            num_hits = list(tar.value).count(1)
            total_num = len(tar)
            print(
                f"According to 'targets' -> num of hits for {l}: {num_hits} out of {len(tar.value)}")
            m = nn.Softmax(dim=0)
            final_out = np.array(m(torch.FloatTensor(raw_out)))
            data_df = pd.DataFrame(list(zip([l] * total_num, raw_out, tar, final_out[:, 1])), columns=[
                                   'label', 'raw_out', 'target', 'prediction'])
            data = pd.concat([data, data_df])
    print(data)
    p = ggplot(aes(x="target", y="prediction"), data=data) + \
        geom_boxplot() + facet_grid(None, "label")
    p.save(figname)


def plot_boxplot(hdf5, epoch=None, figname=None, inverse=False):
    '''
    Plot a boxplot of predictions VS targets useful '
    to visualize the performance of the training algorithm
    This is only usefull in classification tasks

    Args:
        figname (str): filename

    '''

    print('\n --> Box Plot : ', figname, '\n')

    color_plot = {'train': 'red', 'valid': 'blue', 'test': 'green'}
    labels = ['train', 'valid', 'test']

    # -- read data
    h5 = h5py.File(hdf5, 'r')
    if epoch is None:
        keys = list(h5.keys())
        last_epoch_key = list(filter(lambda x: 'epoch_' in x, keys))[-1]
    else:
        last_epoch_key = 'epoch_%04d' % epoch
        if last_epoch_key not in h5:
            print(
                'Incorrect epcoh name\n Possible options are: ' +
                ' '.join(
                    list(
                        h5.keys())))
            h5.close()
            return
    h5data = h5[last_epoch_key]

    print(f"Generate boxplot for {last_epoch_key} epoch ...")

    nwin = len(h5data)

    fig, ax = plt.subplots(1, nwin, sharey=True, squeeze=False)

    iwin = 0
    for l in labels:

        if l in h5data:

            tar = h5data[l]['targets'].value
            out = h5data[l]['outputs'].value

            num_hits = list(tar).count(1)
            print(
                f"According to 'targets' -> num of hits for {l}: {num_hits} out of {len(tar)}")

            data = [[], []]
            for pts, t in zip(out, tar):
                r = F.softmax(torch.FloatTensor(pts), dim=0).data.numpy()
                #print(f"prediction: {pts}; target: {t}; r: {r}")
                data[t].append(r[1])

            ax[0, iwin].boxplot(data)
            ax[0, iwin].set_xlabel(l)
            ax[0, iwin].set_xticklabels(['0', '1'])
            iwin += 1

    fig.savefig(figname, bbox_inches='tight')
    plt.close()


def plot_hit_rate(hdf5, epoch=None, figname=None, inverse=False):
    '''Plot the hit rate of the different training/valid/test sets

    The hit rate is defined as:
        the percentage of positive decoys that are included among the top m decoys.
        a positive decoy is a native-like one with a i-rmsd <= 4A

    Args:
        figname (str): filename for the plot
        irmsd_thr (float, optional): threshold for 'good' models

    '''

    print('\n --> Hit Rate :', figname, '\n')

    color_plot = {'train': 'red', 'valid': 'blue', 'test': 'green'}
    labels = ['train', 'valid', 'test']

    # -- read data
    h5 = h5py.File(hdf5, 'r')
    if epoch is None:
        keys = list(h5.keys())
        last_epoch_key = list(filter(lambda x: 'epoch_' in x, keys))[-1]
    else:
        last_epoch_key = 'epoch_%04d' % epoch
        if last_epoch_key not in h5:
            print(
                'Incorrect epcoh name\n Possible options are: ' +
                ' '.join(
                    list(
                        h5.keys())))
            h5.close()
            return
    data = h5[last_epoch_key]

    print(f"Generate hit rate plot for {last_epoch_key} epoch ...")

    # plot
    fig, ax = plt.subplots()
    for l in labels:
        # l = train, valid or test
        if l in data:
            if 'hit' in data[l]:

                # -- count num_hit
                # hit labels for each model: [0 1 0 0 1...]
                hit_labels = data[l]['hit'].value
                num_hits = list(hit_labels).count(1)
                print(
                    f"According to 'hit' -> num of hits for {l}: {num_hits} out of {len(hit_labels)}")

                # -- calculate and plot hit rate
                hitrate = rankingMetrics.hitrate(data[l]['hit'])
                m = len(hitrate)
                x = np.linspace(0, 100, m)
                plt.plot(x, hitrate, c=color_plot[l], label=l + ' M=%d' % m)
    legend = ax.legend(loc='upper left')
    ax.set_xlabel('Top M (%)')
    ax.set_ylabel('Hit Rate')

    fmt = '%.0f%%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)

    fig.savefig(figname)
    plt.close()


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} epoch_data.hdf5 epoch fig_name")
        sys.exit()
    hdf5 = sys.argv[1]  # 'epoch_data.hdf5'
    epoch = int(sys.argv[2])  # 9
    figname = sys.argv[3]
    plot_hit_rate(
        hdf5,
        epoch=epoch,
        figname=figname +
        '.hitrate.png',
        inverse=False)
    plot_boxplot(
        hdf5,
        epoch=None,
        figname=figname +
        '.boxplot.png',
        inverse=False)
