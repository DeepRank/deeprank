'''
 1. plot prediction scores for class 0 and 1 using two-panel box plots
 2. hit rate plot
 3. success rate plot

 Usage: python {0} epoch_data.hdf5 epoch scenario[all, cm, ranair, refb, ti5, ti] fig_name

'''


import re
import sys
import warnings
from itertools import zip_longest

import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from cal_hitrate_successrate import add_rank, add_perc, ave_evaluate, evaluate, cal_hitrate_successrate
import subprocess
from shlex import quote, split
import pdb

#warnings.filterwarnings("ignore", category=RRuntimeWarning)
USAGE = __doc__.format(sys.argv[0])


def zip_equal(*iterables):
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            # if an element is None
            raise ValueError(f'Iterables have different lengths: {combo}')
        yield combo


def read_epoch_data(DR_h5FL, epoch):
    """# read epoch data into a data frame.

    OUTPUT (pd.DataFrame):

    label               modelID  target        DR                                          sourceFL
    0  Test  1AVX_ranair-it0_5286       0  0.503823  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    1  Test     1AVX_ti5-itw_354w       1  0.502845  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    """
    print (f"-> Read epoch {epoch} data from {DR_h5FL} into df")

    # -- 1. read deeprank output data for the specific epoch
    h5 = h5py.File(DR_h5FL, 'r')

    keys = list(h5.keys())
    last_epoch_key = list(filter(lambda x: 'epoch_' in x, keys))[-1]

    if epoch is None:
        print(f"epoch is not provided. Use the last epoch data: {last_epoch_key}.")
        epoch_key = last_epoch_key
    else:
        epoch_key = 'epoch_%04d' % epoch
        if epoch_key not in h5:
            print('Incorrect epoch name. Use the last epoch data: {last_epoch_key}.')
            epoch_key = last_epoch_key
    data = h5[epoch_key]

    # -- 2. convert into pd.DataFrame
    labels = list(data)  # labels = ['train', 'test', 'valid']

    # write a dataframe of DR and label
    to_plot = pd.DataFrame()
    for l in labels:
        # l = train, valid or test
        source_hdf5FLs = data[l]['mol'][:, 0]
        modelIDs = list(data[l]['mol'][:, 1].astype('str'))
        DR_rawOut = data[l]['outputs']
        DR = F.softmax(torch.FloatTensor(DR_rawOut), dim=1)
        DR = np.array(DR[:, 0])  # the probability of a model being negative

        targets = data[l]['targets'][()]
        targets = targets.astype(np.str)

        to_plot_tmp = pd.DataFrame(
            list(
                zip_equal(
                    source_hdf5FLs,
                    modelIDs,
                    targets,
                    DR)),
            columns=[
                'sourceFL',
                'modelID',
                'target',
                'DR'])
        to_plot_tmp['label'] = l.capitalize()
        to_plot = to_plot.append(to_plot_tmp)

    to_plot['target'] = pd.Categorical(
        to_plot['target'], categories=['0', '1'])
    to_plot['label'] = pd.Categorical(
        to_plot['label'], categories=[
            'Train', 'Valid', 'Test'])

    cols = ['label', 'modelID', 'target', 'DR', 'sourceFL']
    to_plot = to_plot[cols]

    return to_plot


def merge_HS_DR(DR_df, haddockS):
    """INPUT 1 (DR_df: a data frame):

    label    caseID           modelID  target        DR                                          sourceFL
    0  Test  1AVX   1AVX_ranair-it0_5286       0  0.503823  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    1  Test  1AVX   1AVX_ti5-itw_354w       1  0.502845  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    2  Test  1AVX   1AVX_ranair-it0_6223       0  0.511688  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5

    INPUT 2: haddockS[modelID] = score

    OUTPUT (a data frame):

        label caseID   modelID     target   score_method1  score_method2
        Test  1ZHI     1ZHI_294w   0       9.758          -19.3448
        Test  1ZHI     1ZHI_89w    1       17.535         -11.2127
        Train 1ACB     1ACB_9w     1       14.535         -19.2127
    """

    print ("-> Merge HS with DR into one df")

    # -- merge HS with DR predictions, model IDs and class IDs
    modelIDs = DR_df['modelID']
    HS, idx_keep = get_HS(modelIDs, haddockS)

    data = DR_df.iloc[idx_keep, :].copy()
    data['HS'] = HS

    # -- reorder columns
    col_ori = data.columns
    col = ['label', 'caseID', 'modelID', 'target', 'sourceFL']
    col.extend([x for x in col_ori if x not in col])
    data = data[col]

    return data


def read_haddockScoreFL(HS_h5FL):

    print(f"-> Reading haddock score files: {HS_h5FL} ...")

    data = pd.read_hdf(HS_h5FL)

    stats = {}
    stats['haddock-score'] = {}
#    stats['i-RMSD'] = {}

    modelIDs = [re.sub('.pdb', '', x)
                for x in data['modelID']]  # remove .pdb from model ID
    stats['haddock-score'] = dict(zip_equal(modelIDs, data['haddock-score']))
# stats['i-RMSD'] = dict(zip(modelIDs, data['i-RMSD'])) # some i-RMSDs are
# wrong!!! Reported an issue.

    return stats


def plot_DR_iRMSD(df, figname=None):
    """Plot a scatter plot of DeepRank score vs. iRMSD for train, valid and
    test.

    INPUT (a data frame):

        label caseID               modelID target                                          sourceFL        DR      irmsd         HS
        Test   1AVX  1AVX_ranair-it0_5286      0  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5  0.503823  25.189108   6.980802
        Test   1AVX     1AVX_ti5-itw_354w      1  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5  0.502845   3.668682 -95.158100
    """
    print('\n --> Scatter plot of DR vs. iRMSD:', figname, '\n')

   # plot

    font_size = 16
    text_style = element_text(size=font_size, family="Tahoma", face="bold")
    p = ggplot(df) + aes_string(y='irmsd', x='DR') +\
        facet_grid(ro.Formula('.~label')) + \
        geom_point(alpha=0.5) + \
        theme_bw() +\
        theme(**{'plot.title': text_style,
                 'text': text_style,
                 'axis.title': text_style,
                 'axis.text.x': element_text(size=font_size + 2),
                 'axis.text.y': element_text(size=font_size + 2)}) + \
        scale_y_continuous(name="i-RMSD")

    # p.plot()
    ggplot2.ggsave(figname, height=7, width=7 * 1.5, dpi=100)
    return p


def plot_HS_iRMSD(df, figname=None):
    """Plot a scatter plot of HS vs. iRMSD for train, valid and test.

    INPUT (a data frame):

        label caseID               modelID target                                          sourceFL        DR      irmsd         HS
        Test   1AVX  1AVX_ranair-it0_5286      0  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5  0.503823  25.189108   6.980802
        Test   1AVX     1AVX_ti5-itw_354w      1  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5  0.502845   3.668682 -95.158100
    """
    print('\n --> Scatter plot of HS vs. iRMSD:', figname, '\n')

    # plot
    font_size = 16
    text_style = element_text(size=font_size, family="Tahoma", face="bold")
    p = ggplot(df) + aes_string(y='irmsd', x='HS') +\
        facet_grid(ro.Formula('.~label')) + \
        geom_point(alpha=0.5) + \
        theme_bw() +\
        theme(**{'plot.title': text_style,
                 'text': text_style,
                 'axis.title': text_style,
                 'axis.text.x': element_text(size=font_size + 2),
                 'axis.text.y': element_text(size=font_size + 2)}) + \
        scale_y_continuous(name="i-RMSD")

    # p.plot()
    ggplot2.ggsave(figname, height=7, width=7 * 1.5, dpi=100)
    return p


def plot_boxplot(dataFL, figname=None):
    """Plot a boxplot of predictions vs. targets. Useful to visualize the
    performance of the training algorithm. This is only useful in
    classification tasks.

    INPUT (pd.DataFrame):

        label	caseID	modelID	target	DR	HS
        Test	1YVB	1YVB_ranair-it0_4286	0	0.56818	4.04629
        Test	1PPE	1PPE_ranair-it0_2999	0	0.56486	50.17506
    """

    print('\n --> Box Plot: ', figname, '\n')

    command = f'Rscript boxplot.R {dataFL} {figname}'
    print(command)
    command = split(command)
    subprocess.check_call(command)



def plot_successRate_hitRate(df, figname):

    '''
    INPUT:
         label   success_DR  hitRate_DR  success_HS  hitRate_HS  rank      perc
         Test          0.0    0.000000         0.0    0.000000      1  0.000949
         Test          0.0    0.000000         1.0    0.012821      2  0.001898

         Train         0.0    0.000000         1.0    0.012821      1  0.002846
         Train         0.0    0.000000         1.0    0.025641      2  0.003795

    '''


    df.label =pd.Categorical(df.label, categories=["Train","Valid", "Test"])

    # ---------- hit rate plot -------
    figname1 = figname + '.hitRate.png'
    hit_rate_plot(df, figname = figname1)

    # ---------- success rate plot -------
    figname2 = figname + '.successRate.png'
    success_rate_plot(df, figname = figname2)

def hit_rate_plot(df, figname ='hitrate.png'):
    '''
    plot train/valid/test in 3 panels.

    input:
         label   success_dr  hitRate_dr  success_hs  hitRate_hs  rank      perc
         Test          0.0    0.000000         0.0    0.000000      1  0.000949
         Test          0.0    0.000000         1.0    0.012821      2  0.001898

         Train         0.0    0.000000         1.0    0.012821      1  0.002846
         Train         0.0    0.000000         1.0    0.025641      2  0.003795

    '''

    print(f'\n --> hit rate plot:', figname, '\n')

    # -- melt df
    df_melt = pd.melt(df, id_vars=['label', 'rank'])
    idx1 = df_melt.variable.str.contains('^hitRate', case = False)
    df_tmp = df_melt.loc[idx1, :].copy()
    df_tmp.columns = ['label', 'rank', 'Methods', 'hit_rate']

    tmp = list(df_tmp['Methods'])
    df_tmp.loc[:, 'Methods'] = [
        re.sub('hitrate_', '', x) for x in tmp]  # success_dr -> dr

    #-- write to tsv file
    dataFL = 'hitrate_melted.tsv'
    df_tmp.to_csv(dataFL, sep='\t', index = False)
    print(f'{dataFL} generated')

    #-- plot
    command = f'Rscript hitrate_plot.R {dataFL} {figname}'
    print(command)
    command = split(command)
    subprocess.check_call(command)


def success_rate_plot(df, figname):
    """
    INPUT: a pandas data frame
         label   success_DR  hitRate_DR  success_HS  hitRate_HS  rank      perc
         Test          0.0    0.000000         0.0    0.000000      1  0.000949
         Test          0.0    0.000000         1.0    0.012821      2  0.001898

         Train         0.0    0.000000         1.0    0.012821      1  0.002846
         Train         0.0    0.000000         1.0    0.025641      2  0.003795

    """

    print(f'\n --> Success Rate plot:', figname, '\n')

    # -- melt df
    df_melt = pd.melt(df, id_vars=['label', 'rank'])
    idx1 = df_melt.variable.str.contains('^success_', case = False)
    df_tmp = df_melt.loc[idx1, :].copy()
    df_tmp.columns = ['label', 'rank', 'Methods', 'success_rate']

    tmp = list(df_tmp['Methods'])
    df_tmp.loc[:, 'Methods'] = [
        re.sub('success_', '', x) for x in tmp]  # success_DR -> DR

    # -- write to tsv file
    dataFL = 'successrate_melted.tsv'
    df_tmp.to_csv(dataFL, sep = '\t', index = False)
    print(f'{dataFL} generated')

    #-- plot
    command = f'Rscript successrate_plot.R {dataFL} {figname}'
    print(command)
    command = split(command)
    subprocess.check_call(command)

def get_irmsd(source_hdf5, modelIDs):

    irmsd = []
    for h5FL, modelID in zip_equal(source_hdf5, modelIDs):
        # h5FL = '/home/lixue/DBs/BM5-haddock24/hdf5/000_1AY7.hdf5'

        print(modelID)
        f = h5py.File(h5FL, 'r')
        irmsd.append(f[modelID]['targets/IRMSD'][()])
        f.close()
    return irmsd


def get_HS(modelIDs, haddockS):
    HS = []
    idx_keep = []

    # -- remove r000 from modelID
    modelIDs = [re.sub("_r\d+$",'', modelID) for modelID in modelIDs ]

    # -- retrieve HS
    for idx, modelID in enumerate(modelIDs):
        if modelID in haddockS:
            HS.append(haddockS[modelID])
            idx_keep.append(idx)
        else:
            print(f"Warning: model ID {modelID} does not have HS.")
    return HS, idx_keep

def filter_models(df, label = 'Test', scenario = 'ranair'):
    '''
    Keep a subset of models for the ploting, e.g, ab-initio models in the Test set.

    INPUT (pd.dataframe):
    df:
       label               modelID  target        DR                                          sourceFL
       Test  1AVX_ranair-it0_5286       0  0.503823  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
       Test     1AVX_ti5-itw_354w       1  0.502845  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5

    OUTPUT (pd.dataframe):
    df:
       label               modelID  target        DR                                          sourceFL
       Test  1AVX_ranair-it0_5286       0  0.503823  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5

    '''

    print (f"-> Keep models for {scenario} in the {label} set")
    idx1 = df.label == label
    idx2 = df.modelID.str.contains( scenario , case = False)
    throw = idx1 & ~idx2
    df = df[~throw]
    return df

def add_irmsd(df):
    '''
    INPUT (a data frame):
    df:
        label caseID   modelID                 sourceFL          target   score_method1  score_method2
        train 1ZHI     1ZHI_294w  ..../hdf5/000_1ZHI.hdf5        0       9.758          -19.3448
        test  1ACB     1ACB_89w   ..../hdf5/000_1ACB.hdf5        1       17.535         -11.2127

    OUTPUT (a data frame):
    df:
        label    caseID   modelID     irmsd    target   score_method1  score_method2
        train     1ZHI     1ZHI_294w   12.1    0       9.758          -19.3448
        train     1ZHI     1ZHI_89w    1.3     1       17.535         -11.2127
        ...
        test      1ACB     1ACB_89w    2.4     1       17.535         -11.2127
    '''

    print ("-> Add i-RMSD to df")

    modelIDs = df['modelID']
    source_hdf5FLs = df['sourceFL']
    irmsd = np.array(get_irmsd(source_hdf5FLs, modelIDs))
    df['irmsd'] = irmsd
    return df

def add_caseID(df):

    ''' add the caseID column.

    INPUT (a data frame):

    label               modelID  target        DR                                          sourceFL
    0  Test  1AVX_ranair-it0_5286       0  0.503823  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    1  Test     1AVX_ti5-itw_354w       1  0.502845  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    '''

    df['caseID'] = [re.split('_', x)[0] for x in df['modelID']]
    return df


def remove_failedCases(df):

    '''
    Remove cases with any hits.

    INPUT (a data frame):

    label   caseID             modelID  target        DR                                          sourceFL
    0  Test 1AVX  1AVX_ranair-it0_5286       0  0.503823  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    1  Test 1AVX     1AVX_ti5-itw_354w       1  0.502845  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    2  Test 1AVX  1AVX_ranair-it0_6223       0  0.511688  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    '''

    grp = df.groupby('caseID')
    for caseID, df_ in grp:


        num_hits = sum(df_.target.astype('int'))
        if num_hits ==0:
            print(f"case {caseID} does not have any hits and is removed.")
            idx = list(df_.index)
            df.drop(idx, axis = 0)

    return df

def prepare_df(deeprank_h5FL, HS_h5FL, epoch, scenario):
    '''
    OUTPUT: a data frame:

        label caseID               modelID target                                          sourceFL        DR      irmsd         HS
        Test   1AVX  1AVX_ranair-it0_5286      0  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5  0.503823  25.189108   6.980802
        Test   1AVX     1AVX_ti5-itw_354w      1  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5  0.502845   3.668682 -95.158100

    '''

    print ("=== Prepare the df ===")
    # -- read deeprank_h5FL epoch data into pd.DataFrame (DR_df)
    DR_df = read_epoch_data(deeprank_h5FL, epoch)

    '''
    DR_df (a data frame):

    label               modelID  target        DR                                          sourceFL
    0  Test  1AVX_ranair-it0_5286       0  0.503823  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    1  Test     1AVX_ti5-itw_354w       1  0.502845  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    2  Test  1AVX_ranair-it0_6223       0  0.511688  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    '''

    #-- keep subset of models
    if scenario != 'all':
        DR_df = filter_models(DR_df, label = 'Valid', scenario= scenario )
        DR_df = filter_models(DR_df, label = 'Test', scenario= scenario )

    # -- remove cases with any hits
    DR_df = add_caseID(DR_df)
    DR_df = remove_failedCases(DR_df)

    # -- add iRMSD column to DR_df
#    DR_df = add_irmsd(DR_df)


    # -- add HS to DR_df (note: bound complexes do not have HS)
    stats = read_haddockScoreFL(HS_h5FL)
    haddockS = stats['haddock-score']  # haddockS[modelID] = score
    DR_HS_df = merge_HS_DR(DR_df, haddockS)

    '''
    DR_HS_df (a data frame):

    data:
        label caseID   modelID                                          sourceFL          target   score_method1  score_method2
        train 1ZHI     1ZHI_294w  /home/lixue/DBs/BM5-haddock24/hdf5/000_1ZHI.hdf5        0       9.758          -19.3448
        test  1ACB     1ACB_89w   /home/lixue/DBs/BM5-haddock24/hdf5/000_1ACB.hdf5        1       17.535         -11.2127
    '''

    DR_HS_df['label'] = pd.Categorical(DR_HS_df['label'], categories=[
                                 'Train', 'Valid', 'Test'])

    print ("\n === df preparation done. === \n")

    return DR_HS_df


def hit_statistics(df):
    """Report the number of hits for Train, valid and test.

    INPUT (a data frame):

        label               modelID target        DR                                          sourceFL      irmsd
        Test  1AVX_ranair-it0_5286      0  0.503823  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5  25.189108
        Test     1AVX_ti5-itw_354w      1  0.502845  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5   3.668682
    """

    df['target'] = df['target'].astype(int)

    grouped = df.groupby('label')


#    # -- 1. count num_hit based on i-rmsd
#    num_hits = grouped['irmsd'].apply(lambda x: len(x[x <= 4]))
#    num_models = grouped.apply(len)
#
#    for label, _ in grouped:
#        print(
#            f"According to 'i-RMSD' -> num of hits for {label}: {num_hits[label]} out of {num_models[label]} models")
#
#    print("")

    # -- 2. count num_hit based on the 'target' column
    num_hits = grouped['target'].apply(lambda x: len(x[x == 1]))
    num_models = grouped.apply(len)

    for label, _ in grouped:
        print(f"According to 'targets' -> num of hits for {label}: {num_hits[label]} out of {num_models[label]} models")

    print("")

    # -- 3. report num of cases without hits
    df_tmp = df.loc[:,['label','modelID', 'target']]
    df_tmp['caseID'] = df['modelID'].apply(get_caseID)
    grp1 = df_tmp.groupby(['label', 'caseID'])

    num_hits = grp1['target'].apply(lambda x: sum(x)) # the number of hits for each case
    grp2 = num_hits.groupby('label')
    num_cases_total = grp2.apply(len)
    num_cases_wo_hit = grp2.apply(lambda x: len(x[x == 0]))

    for label, _ in grouped:
        print(
                f"According to 'targets' -> num of cases w/o hits for {label}: {num_cases_wo_hit[label]} out of {num_cases_total[label]} cases")
    print("")

    #-- 4. report num of hits for each label
    grp = df.groupby('label')
    hit_stats = grp['target'].agg([numHits, numModels])
    print(hit_stats)

    grp = df.groupby(['label', 'caseID'])
    hit_stats2 = grp['target'].agg([numHits, numModels])
    hit_stats2.to_csv('hit_stats.tsv', sep = '\t')
    print('hit_stats.tsv generated\n')

def numHits(target):
    # target = [0, 0, 1, 0,... ]
    num_hits = sum(target.astype(int))
    return num_hits

def numModels(target):
    # target = [0, 0, 1, 0,... ]
    num_models = len(target)
    return num_models

def get_caseID(modelID):
    # modelID = 1AVX_ranair-it0_5286
    # caseID = 1AVX

    tmp = re.split('_', modelID)
    caseID = tmp[0]
    return caseID

#-------------------------------------
#--- BEGIN: functions not used -------

def plot_boxplot_r2py(df, figname=None, inverse=False):
    """Plot a boxplot of predictions vs. targets. Useful to visualize the
    performance of the training algorithm. This is only useful in
    classification tasks.

    INPUT (pd.DataFrame):

       label               modelID  target        DR                                          sourceFL
       Test  1AVX_ranair-it0_5286       0  0.503823  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
       Test     1AVX_ti5-itw_354w       1  0.502845  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
       Test  1AVX_ranair-it0_6223       0  0.511688  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    """

    pandas2ri.activate()

    print('\n --> Box Plot: ', figname, '\n')

    data = df

    font_size = 20
    # line = "#1F3552"

    text_style = element_text(size=font_size, family="Tahoma", face="bold")

    colormap_raw = [['0', 'ivory3'],
                    ['1', 'steelblue']]

    colormap = ro.StrVector([elt[1] for elt in colormap_raw])
    colormap.names = ro.StrVector([elt[0] for elt in colormap_raw])

    p = ggplot(data) + \
        aes_string(x='target', y='DR', fill='target') + \
        geom_boxplot(width=0.2, alpha=0.7) + \
        facet_grid(ro.Formula('.~label')) +\
        scale_fill_manual(values=colormap) + \
        theme_bw() +\
        theme(**{'plot.title': text_style,
                 'text': text_style,
                 'axis.title': text_style,
                 'axis.text.x': element_text(size=font_size),
                 'legend.position': 'right'}) +\
        scale_x_discrete(name="Target")

    # p.plot()
    ggplot2.ggsave(figname, dpi=100)
    return p



def hit_rate_plot_r2py(df, figname ='hitrate.png'):
    '''
    plot train/valid/test in 3 panels.

    input:
         label   success_dr  hitrate_dr  success_hs  hitrate_hs  rank      perc
         test          0.0    0.000000         0.0    0.000000      1  0.000949
         test          0.0    0.000000         1.0    0.012821      2  0.001898

         train         0.0    0.000000         1.0    0.012821      1  0.002846
         train         0.0    0.000000         1.0    0.025641      2  0.003795

    '''
    pandas2ri.activate()
    print(f'\n --> hit rate plot:', figname, '\n')

    # -- melt df
    df_melt = pd.melt(df, id_vars=['label', 'rank'])
    idx1 = df_melt.variable.str.contains('^hitrate', case = False)
    df_tmp = df_melt.loc[idx1, :].copy()
    df_tmp.columns = ['label', 'rank', 'methods', 'hit_rate']

    tmp = list(df_tmp['methods'])
    df_tmp.loc[:, 'methods'] = [
        re.sub('hitrate_', '', x) for x in tmp]  # success_dr -> dr

    font_size = 40
    breaks = pd.to_numeric(np.arange(0, 1.01, 0.25))
    #xlabels = list(map(lambda x: str('%d' % (x * 100)) +' % ', np.arange(0, 1.01, 0.25)))
    text_style = element_text(size=font_size, family="tahoma", face="bold")

    df_tmp.to_csv('hitrate_melted.tsv', sep='\t', index = false)
    print('hitrate_melted.tsv generated')
    p = ggplot(df_tmp) + \
        aes_string(x='rank', y='hit_rate', color='label', linetype='methods') + \
        facet_grid(ro.formula("label~.")) +\
        geom_line(size=1) + \
        labs(**{'x': 'top n models', 'y': 'hit rate'}) + \
        theme_bw() + \
        theme(**{
                 'legend.position': 'right',
                 'plot.title': text_style,
                 'text': text_style,
                 'axis.text.x': element_text(size=font_size),
                 'axis.text.y': element_text(size=font_size)}) +\
        labs(**{'colour': "sets"}) #change legend title to 'sets'

    # scale_x_continuous(**{'breaks': breaks, 'labels': xlabels})

    ggplot2.ggsave(figname, height=7*3, width=7 * 1.2*3, dpi=50)

def success_rate_plot_r2py(df):
    """
    INPUT: a pandas data frame
         label   success_DR  hitRate_DR  success_HS  hitRate_HS  rank      perc
         Test          0.0    0.000000         0.0    0.000000      1  0.000949
         Test          0.0    0.000000         1.0    0.012821      2  0.001898

         Train         0.0    0.000000         1.0    0.012821      1  0.002846
         Train         0.0    0.000000         1.0    0.025641      2  0.003795

    """

    pandas2ri.activate()

    # -- melt df
    df_melt = pd.melt(df, id_vars=['label', 'rank'])
    idx1 = df_melt.variable.str.contains('^success_', case = False)
    df_tmp = df_melt.loc[idx1, :].copy()
    df_tmp.columns = ['label', 'rank', 'Methods', 'success_rate']

    tmp = list(df_tmp['Methods'])
    df_tmp.loc[:, 'Methods'] = [
        re.sub('success_', '', x) for x in tmp]  # success_DR -> DR

    font_size = 40
#    breaks = pd.to_numeric(np.arange(0, 1.01, 0.25))
#    xlabels = list(map(lambda x: str('%d' % (x * 100)) +
#                       ' % ', np.arange(0, 1.01, 0.25)))
    text_style = element_text(size=font_size, family="Tahoma", face="bold")

    df_tmp.to_csv('successrate_melted.tsv', sep='\t', index = False)
    print('successrate_melted.tsv generated')

    p = ggplot(df_tmp) + \
        aes_string(x='rank', y='success_rate', color='label', linetype='Methods') + \
        facet_grid(ro.Formula('label ~.')) +\
        geom_line(size=1) + \
        labs(**{'x': 'Top N models', 'y': 'Success Rate'}) + \
        theme_bw() + \
        theme(**{'legend.position': 'right',
                 'plot.title': text_style,
                 'text': text_style,
                 'axis.text.x': element_text(size=font_size),
                 'axis.text.y': element_text(size=font_size)}) +\
        labs(**{'colour': "Sets"}) #change legend title to 'Sets'
#        scale_x_continuous(**{'breaks': breaks, 'labels': xlabels})

    ggplot2.ggsave(figname, height=7*3, width=7 * 1.2*3, dpi=50)

#-----------------------------------
#--- END: functions not used -------


#def main(HS_h5FL='/home/lixue/DBs/BM5-haddock24/stats/stats.h5'): # on alembick
def main(HS_h5FL='/projects/0/deeprank/BM5/docked_models/stats.h5'): # on cartesius
    if len(sys.argv) != 5:
        sys.exit(USAGE)
    # the output h5 file from deeprank: 'epoch_data.hdf5'
    deeprank_h5FL = sys.argv[1]
    epoch = int(sys.argv[2])  # 9
    scenario = sys.argv[3] # cm, ranair, refb, ti5, ti, or it0, it1, itw or other patterns in the modelID
    figname = sys.argv[4]

    #-- read deeprank.hdf5 and HS.hdf5 to a pandas df
    df = prepare_df(deeprank_h5FL, HS_h5FL, epoch, scenario)
    rawdataFL=f'{scenario}.rawdata.tsv'
    df.to_csv(rawdataFL, sep = '\t', index = False, float_format = '%.5f')
    print(f'{rawdataFL} generated.\n')

    #rawdataFL=f'{scenario}.rawdata.tsv'
    #df = pd.read_csv(rawdataFL, sep='\t')

    # -- report the number of hits for train/valid/test
    hit_statistics(df)

    #-- calculate hit rate and success rate
    hitrate_successrate_per_case, hitrate_successrate_df = cal_hitrate_successrate(df[['label',
                                 'caseID',
                                 'modelID',
                                 'target',
                                 'DR',
                                 'HS']].copy())
    #-- plot
    plot_successRate_hitRate(hitrate_successrate_df, figname=f"{figname}.epo{epoch}.{scenario}")
    plot_boxplot(rawdataFL, figname=f"{figname}.epo{epoch}.{scenario}.boxplot.png")

    #--
    #Note: plot_HS_iRMSD and plot_DR_iRMSD disabled due to the long running time.
    #    plot_HS_iRMSD(df, figname=f"{figname}.epo{epoch}.{scenario}.irsmd_HS.png")
    #    plot_DR_iRMSD(df, figname=f"{figname}.epo{epoch}.{scenario}.irsmd_HS.png")
    #--



if __name__ == '__main__':
    main()
