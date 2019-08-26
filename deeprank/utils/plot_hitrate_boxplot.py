# 1. plot prediction scores for class 0 and 1 using two-panel box plots
# 2. plot hit rate plot
import glob
import os
import pdb
import re
import sys

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from deeprank.learn import rankingMetrics
from ggplot import *


def plot_boxplot_todo(hdf5, epoch=None, figname=None, inverse=False):
    """Plot a boxplot of predictions VS targets useful ' to visualize the
    performance of the training algorithm This is only usefull in
    classification tasks.

    Args:
        figname (str): filename
    """

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


def sort_modelIDs_by_deeprank(modelIDs, deeprank_score):
    out = F.softmax(
        torch.FloatTensor(deeprank_score),
        dim=1).data.numpy()[
        :,
        1]
#    modelIDs_sorted = [y for x, y in sorted(zip(out,modelIDs))]
#    modelIDs_sorted = modelIDs_sorted[::-1] #reverse the list

    xue = pd.DataFrame(list(zip(modelIDs, out)),
                       columns=['modelID', 'final_S'])
    xue_sorted = xue.sort_values(by='final_S', ascending=False)
    modelIDs_sorted = list(xue_sorted['modelID'])
    return modelIDs_sorted


def plot_boxplot(hdf5, epoch=None, figname=None, inverse=False):
    """Plot a boxplot of predictions VS targets useful ' to visualize the
    performance of the training algorithm This is only usefull in
    classification tasks.

    Args:
        figname (str): filename
    """

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


def plot_hit_rate_withHS(
        hdf5,
        HS_DIR=None,
        epoch=None,
        figname=None,
        inverse=False):
    """Plot the hit rate of the different training/valid/test sets with HS
    (haddock scores)

    The hit rate is defined as:
        the percentage of positive decoys that are included among the top m decoys.
        a positive decoy is a native-like one with a i-rmsd <= 4A

    Args:
        HS_DIR (str): the directory where HS files are stored
        figname (str): filename for the plot
        irmsd_thr (float, optional): threshold for 'good' models
    """

    print('\n --> Hit Rate :', figname, '\n')

    color_plot = {
        'train': 'red',
        'valid': 'blue',
        'test': 'green',
        'HS-train': 'red',
        'HS-valid': 'blue',
        'HS-test': 'green'}
    line_styles = {
        'train': '-',
        'valid': '-',
        'test': '-',
        'HS-train': '--',
        'HS-valid': '--',
        'HS-test': '--'}
    labels = ['train', 'valid', 'test', 'HS']

    # -- read haddock data
    stats = read_haddockScoreFL(HS_DIR)
    haddockS = stats['haddock-score']  # haddockS[modelID] = score

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

                # -- calculate and plot hit rate)for haddock
                #                pdb.set_trace()
                # np.ndarray, hit labels for each model: [0 1 0 0 1...]
                hit_labels_deeprank = data[l]['hit'].value
                modelIDs_deeprank = sort_modelIDs_by_deeprank(list(
                    data[l]['mol'][:, 1]), data[l]['outputs'])  # np.ndarry, models IDs ranked by deeprank

#                xue = pd.DataFrame(list(zip(modelIDs_deeprank, hit_labels_deeprank)), columns=['modelIDs_DR', 'hit_labels_DR'])
#                xue.to_csv('xue.tsv', sep="\t")
                [hit_labels_HS, modelIDs_woHS] = get_hit_labels_HS(
                    haddockS, hit_labels_deeprank, modelIDs_deeprank)
                hitrate_HS = rankingMetrics.hitrate(hit_labels_HS)

                m = len(hitrate_HS)
                x = np.linspace(0, 100, m)
                legend = 'HS-' + l
                print(f"legend:{legend}")
                ax.plot(
                    x,
                    hitrate_HS,
                    color=color_plot[legend],
                    linestyle=line_styles[legend],
                    label=f'{legend}' +
                    ' M=%d' %
                    m)
#                pdb.set_trace()

                # -- remove refe pdb from hit rate calcuatioin as we do not have HS for refe
                print(f"Models w/o haddock scores: {modelIDs_woHS}.")
                print(f"Now remove them from calculating hit rate!")
                indices_to_remove = [
                    modelIDs_deeprank.index(x) for x in modelIDs_woHS]
                hit_labels_deeprank = np.delete(
                    hit_labels_deeprank, indices_to_remove)

                # -- calculate and plot hit rate for deeprank
#                hitrate_deeprank = rankingMetrics.hitrate(data[l]['hit'])
                hitrate_deeprank = rankingMetrics.hitrate(hit_labels_deeprank)
                m = len(hitrate_deeprank)
                x = np.linspace(0, 100, m)
                ax.plot(
                    x,
                    hitrate_deeprank,
                    c=color_plot[l],
                    linestyle=line_styles[l],
                    label=l +
                    ' M=%d' %
                    m)

                # -- count num_hit
                num_hits = list(hit_labels_deeprank).count(1)
                print(
                    f"According to 'hit' -> num of hits for {l}: {num_hits} out of {len(hit_labels_deeprank)}")

                # -- write to csv file
                # pdb.set_trace()

    legend = ax.legend(loc='upper left')
    ax.set_xlabel('Top M (%)')
    ax.set_ylabel('Hit Rate')

    fmt = '%.0f%%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)

    fig.savefig(figname)
    plt.close()


def get_hit_labels_HS(haddockS, hit_labels_deeprank, modelIDs_deeprank):
    # reorder hit_labels_deeprank based on haddock scores

    HS = []
    modelIDs_woHS = []
    modelIDs_HS = []

    for modelID in modelIDs_deeprank:
        if modelID in haddockS:
            HS.append(haddockS[modelID])
            modelIDs_HS.append(modelID)
        else:
            modelIDs_woHS.append(modelID)

  # -- remove refe pdb from hit rate calcuatioin as we do not have HS for refe
    print(f"Models w/o haddock scores: {modelIDs_woHS}.")
    print(f"Now remove them from calculating hit rate!")
    indices_to_remove = [modelIDs_deeprank.index(x) for x in modelIDs_woHS]
    hit_labels_deeprank = np.delete(hit_labels_deeprank, indices_to_remove)

    data = pd.DataFrame(list(zip(modelIDs_HS, HS, hit_labels_deeprank)), columns=[
                        'modelID', 'HS', 'hit_labels'])
    data_sorted = data.sort_values(by='HS', ascending=True)
    hit_labels_HS = data_sorted['hit_labels']

    return hit_labels_HS, modelIDs_woHS


def read_haddockScoreFL(HS_DIR):
    '''
    input: str. /home/lixue/DBs/BM5-haddock24/stats
    output: dict. stats['haddock-score'][modelID] = score

    stat file format:

    #struc haddock-score i-RMSD Einter Enb Evdw+0.1Eelec Evdw Eelec Eair Ecdih Ecoup Esani Evean Edani #NOEviol #Dihedviol #Jviol #Saniviol #veanviol #Daniviol bsa dH Edesolv
    1A2K_cm-itw_31w.pdb -124.227921 18.868 -322.996 -323.353 -98.3385 -73.3369 -250.016 0.356572 0 0 0 0 0 0 0 0 0 0 0 2094.49 -30.0497 -0.887821
    1A2K_cm-itw_187w.pdb -123.982600 18.968 -383.472 -384.327 -76.94 -42.7859 -341.541 0.855228 0 0 0 0 0 0 0 0 0 0 0 1671.79 -71.7494 -12.8885

    '''

    stat_FLs = glob.glob(f"{HS_DIR}/*.stats")
    stats = {}

    stat_FLs = tqdm(
        stat_FLs,
        desc='read stat files for haddock scores',
        disable=False)
    for statFL in stat_FLs:

        f = open(statFL, 'r')
        for line in f:

            line = line.rstrip()
            line = line.strip()

            if re.search('^#', line):
                headers = re.split(r'\s+', line)
                headers = headers[1:]
                continue
            values = re.split(r'\s+', line)
            modelID = re.sub('.pdb', '', values.pop(0))

            if len(headers) != len(values):
                sys.exit(
                    f'header field number {len(headers)} is different from the value field number {len(values)}. Check the format of {statFL}')

            for idx, h in enumerate(headers):
                if h not in stats:
                    stats[h] = {}

                stats[h][modelID] = float(values[idx])

        if not stats or not headers or not values:
            sys.exit(
                f"headers or values or stats not defined. Check the format of {statFL}")
        f.close()

    return stats


def plot_hit_rate(hdf5, epoch=None, figname=None, inverse=False):
    """Plot the hit rate of the different training/valid/test sets.

    The hit rate is defined as:
        the percentage of positive decoys that are included among the top m decoys.
        a positive decoy is a native-like one with a i-rmsd <= 4A

    Args:
        figname (str): filename for the plot
        irmsd_thr (float, optional): threshold for 'good' models
    """

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


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {python sys.argv[0]} epoch_data.hdf5 epoch fig_name")
        sys.exit()
    hdf5 = sys.argv[1]  # 'epoch_data.hdf5'
    epoch = int(sys.argv[2])  # 9
    figname = sys.argv[3]
    plot_hit_rate_withHS(
        hdf5,
        HS_DIR='/home/lixue/DBs/BM5-haddock24/stats',
        epoch=epoch,
        figname=figname +
        '.hitrate_wHS.png',
        inverse=False)
    plot_hit_rate(
        hdf5,
        epoch=epoch,
        figname=figname +
        '.hitrate.png',
        inverse=False)
    plot_boxplot(
        hdf5,
        epoch=epoch,
        figname=figname +
        '.boxplot.png',
        inverse=False)


if __name__ == '__main__':
    main()
