# 1. plot prediction scores for class 0 and 1 using two-panel box plots
# 2. hit rate plot
# 3. success rate plot
from deeprank.learn import rankingMetrics
import numpy as np
import h5py
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import re
from itertools import zip_longest

import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)

from rpy2.robjects.lib.ggplot2 import *
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro


def zip_equal(*iterables):
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            # if an element is None
            raise ValueError(f'Iterables have different lengths: {combo}')
        yield combo


def plot_boxplot(df,figname=None,inverse = False):

    '''
    Plot a boxplot of predictions vs. targets. Useful
    to visualize the performance of the training algorithm.
    This is only useful in classification tasks.

    INPUT (pd.DataFrame):

       label               modelID  target        DR                                          sourceFL
       Test  1AVX_ranair-it0_5286       0  0.503823  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
       Test     1AVX_ti5-itw_354w       1  0.502845  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
       Test  1AVX_ranair-it0_6223       0  0.511688  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5

    '''

    print('\n --> Box Plot : ', figname, '\n')

    data = df

    font_size = 20
    #line = "#1F3552"

    text_style = element_text(size = font_size, family = "Tahoma", face = "bold")

    colormap_raw =[['0','ivory3'],
            ['1','steelblue']]

    colormap = ro.StrVector([elt[1] for elt in colormap_raw])
    colormap.names = ro.StrVector([elt[0] for elt in colormap_raw])

    p= ggplot(data) + \
            aes_string(x='target', y='DR' , fill='target' ) + \
            geom_boxplot( width = 0.2, alpha = 0.7) + \
            facet_grid(ro.Formula('.~label')) +\
            scale_fill_manual(values = colormap ) + \
            theme_bw() +\
            theme(**{'plot.title' : text_style,
                'text':  text_style,
                'axis.title':  text_style,
                'axis.text.x': element_text(size = font_size),
                'legend.position': 'right'} ) +\
            scale_x_discrete(name = "Target")

    # p.plot()
    ggplot2.ggsave(figname, dpi = 100)
    return p


def read_epoch_data(DR_h5FL, epoch):
    '''
    # read epoch data into a data frame

    OUTPUT (pd.DataFrame):

    label               modelID  target        DR                                          sourceFL
    0  Test  1AVX_ranair-it0_5286       0  0.503823  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    1  Test     1AVX_ti5-itw_354w       1  0.502845  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    '''

    #-- 1. read deeprank output data for the specific epoch
    h5 = h5py.File(DR_h5FL,'r')
    if epoch is None:
        print (f"epoch is not provided. Use the last epoch data.")
        keys = list(h5.keys())
        last_epoch_key = list(filter(lambda x: 'epoch_' in x,keys))[-1]
    else:
        last_epoch_key = 'epoch_%04d' %epoch
        if last_epoch_key not in h5:
            print('Incorrect epcoh name\n Possible options are: ' + ' '.join(list(h5.keys())))
            h5.close()
            return
    data = h5[last_epoch_key]


    #-- 2. convert into pd.DataFrame
    labels = list(data) # labels = ['train', 'test', 'valid']

    # write a dataframe of DR and label
    to_plot = pd.DataFrame()
    for l in labels:
        # l = train, valid or test
        source_hdf5FLs = data[l]['mol'][:,0]
        modelIDs = list(data[l]['mol'][:,1])
        DR_rawOut = data[l]['outputs']
        DR = F.softmax(torch.FloatTensor(DR_rawOut), dim = 1)
        DR = np.array(DR[:,0]) # the probability of a model being negative

        targets =  data[l]['targets'][()]
        targets = targets.astype(np.str)

        to_plot_tmp = pd.DataFrame(list(zip_equal(source_hdf5FLs, modelIDs, targets, DR)), columns = ['sourceFL', 'modelID', 'target', 'DR'])
        to_plot_tmp['label'] = l.capitalize()
        to_plot = to_plot.append(to_plot_tmp)

    to_plot['target'] = pd.Categorical(to_plot['target'], categories=['0', '1'])
    to_plot['label'] = pd.Categorical(to_plot['label'], categories=['Train', 'Valid', 'Test'])

    cols = ['label', 'modelID', 'target', 'DR', 'sourceFL']
    to_plot = to_plot[cols]


    return to_plot

def merge_HS_DR(DR_df, haddockS):

    '''
    INPUT 1 (DR_df: a data frame):

    label               modelID  target        DR                                          sourceFL
    0  Test  1AVX_ranair-it0_5286       0  0.503823  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    1  Test     1AVX_ti5-itw_354w       1  0.502845  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    2  Test  1AVX_ranair-it0_6223       0  0.511688  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5

    INPUT 2: haddockS[modelID] = score

    OUTPUT (a data frame):

        label caseID   modelID     target   score_method1  score_method2
        Test  1ZHI     1ZHI_294w   0       9.758          -19.3448
        Test  1ZHI     1ZHI_89w    1       17.535         -11.2127
        Train 1ACB     1ACB_9w     1       14.535         -19.2127
    '''


    #-- merge HS with DR predictions, model IDs and class IDs
    modelIDs = DR_df['modelID']
    HS, idx_keep = get_HS(modelIDs, haddockS)

    data = DR_df.iloc[idx_keep,:].copy()
    data['HS'] = HS
    data['caseID'] = [re.split('_', x)[0] for x in data['modelID']]


    #-- reorder columns
    col_ori = data.columns
    col = ['label', 'caseID', 'modelID', 'target', 'sourceFL']
    col.extend( [x for x in col_ori if x not in col])
    data = data[col]

    return data


def read_haddockScoreFL(HS_h5FL):

    print(f"Reading haddock score files: {HS_h5FL} ...")
    data = pd.read_hdf(HS_h5FL)

    stats = {}
    stats['haddock-score'] = {}
#    stats['i-RMSD'] = {}

    modelIDs = [ re.sub('.pdb','',x) for x in data['modelID'] ] # remove .pdb from model ID
    stats['haddock-score'] = dict(zip_equal(modelIDs, data['haddock-score']))
#    stats['i-RMSD'] = dict(zip(modelIDs, data['i-RMSD'])) # some i-RMSDs are wrong!!! Reported an issue.

    return stats

def plot_DR_iRMSD(df, figname=None):
    '''
    Plot a scatter plot of DeepRank score vs. iRMSD for train, valid and test

    INPUT (a data frame):

        label caseID               modelID target                                          sourceFL        DR      irmsd         HS
        Test   1AVX  1AVX_ranair-it0_5286      0  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5  0.503823  25.189108   6.980802
        Test   1AVX     1AVX_ti5-itw_354w      1  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5  0.502845   3.668682 -95.158100

    '''
    print('\n --> Scatter plot of DR vs. iRMSD:', figname, '\n')

   # plot

    font_size = 16
    text_style = element_text(size = font_size, family = "Tahoma", face = "bold")
    p = ggplot(df) + aes_string(y = 'irmsd', x = 'DR') +\
            facet_grid(ro.Formula('.~label')) + \
            geom_point(alpha = 0.5) + \
            theme_bw() +\
            theme(**{'plot.title' : text_style,
                    'text':  text_style,
                    'axis.title':  text_style,
                    'axis.text.x': element_text(size = font_size + 2),
                    'axis.text.y': element_text(size = font_size + 2)} ) + \
            scale_y_continuous(name = "i-RMSD")

    #p.plot()
    ggplot2.ggsave(figname, height = 7 , width = 7 * 1.5, dpi = 100)
    return p



def plot_HS_iRMSD(df, figname=None):
    '''
    Plot a scatter plot of HS vs. iRMSD for train, valid and test

    INPUT (a data frame):

        label caseID               modelID target                                          sourceFL        DR      irmsd         HS
        Test   1AVX  1AVX_ranair-it0_5286      0  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5  0.503823  25.189108   6.980802
        Test   1AVX     1AVX_ti5-itw_354w      1  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5  0.502845   3.668682 -95.158100

    '''
    print('\n --> Scatter plot of HS vs. iRMSD:', figname, '\n')

    # plot
    font_size = 16
    text_style = element_text(size = font_size, family = "Tahoma", face = "bold")
    p= ggplot(df) + aes_string(y = 'irmsd', x = 'HS') +\
            facet_grid(ro.Formula('.~label')) + \
            geom_point(alpha = 0.5) + \
            theme_bw() +\
            theme(**{'plot.title' : text_style,
                    'text':  text_style,
                    'axis.title':  text_style,
                    'axis.text.x': element_text(size = font_size + 2),
                    'axis.text.y': element_text(size = font_size + 2)} ) + \
            scale_y_continuous(name = "i-RMSD")

    #p.plot()
    ggplot2.ggsave(figname, height = 7 , width = 7 * 1.5, dpi=100)
    return p


def plot_successRate_hitRate (df, figname=None,inverse = False):
    '''Plot the hit rate and success_rate of the different training/valid/test sets with HS (haddock scores)

    The hit rate is defined as:
        the percentage of positive decoys that are included among the top m decoys.
        a positive decoy is a native-like one with a i-rmsd <= 4A

    Args:
        DR_h5FL (str): the hdf5 file generated by DeepRank.
        HS_h5FL (str): the hdf5 file that saves data from haddock *.stats files
        figname (str): filename for the plot

    Steps:
    0. Input data:

        label caseID               modelID target        DR         HS
        0  Test   1AVX  1AVX_ranair-it0_5286      0  0.503823   6.980802
        1  Test   1AVX     1AVX_ti5-itw_354w      1  0.502845 -95.158100
        2  Test   1AVX  1AVX_ranair-it0_6223      0  0.511688 -11.961460


    1. For each case, calculate hit rate and success. Success is a binary, indicating whether this case is success when evaluating its top N models.

            caseID   success_DR   hitRate_DR   success_HS   hitRate_HS
            1ZHI     1            0.1          0            0.01
            1ZHI     1            0.2          1            0.3
            ...

            1ACB     0            0            1            0.3
            1ACB     1            0.2          1            0.4
            ...
    2. Calculate success rate and hit rate over all cases.


    '''

    #-- 1. calculate success rate and hit rate
    performance_per_case = evaluate(df)
    performance_ave = ave_evaluate(performance_per_case)
    performance_ave = add_rank(performance_ave)

    #-- 2. plot
    plot_evaluation(performance_ave, figname)

def add_rank(df):
    '''
    INPUT (a data frame):
         label   success_DR  hitRate_DR  success_HS  hitRate_HS
         Test          0.0    0.000000         0.0    0.000000
         Test          0.0    0.000000         1.0    0.012821

         Train         0.0    0.000000         1.0    0.012821
         Train         0.0    0.000000         1.0    0.025641

    OUTPUT:
         label   success_DR  hitRate_DR  success_HS  hitRate_HS      rank
         Test          0.0    0.000000         0.0    0.000000  0.000949
         Test          0.0    0.000000         1.0    0.012821  0.001898

         Train         0.0    0.000000         1.0    0.012821  0.002846
         Train         0.0    0.000000         1.0    0.025641  0.003795

    '''

    #-- add the 'rank' column to df
    rank = []
    for l, df_per_label in df.groupby('label'):
        num_mol = len(df_per_label)
        rank_raw = np.array(range(num_mol )) + 1
        rank.extend(rank_raw/num_mol )
    df['rank'] = rank

    df['label'] = pd.Categorical(df['label'], categories=['Train', 'Valid', 'Test'])

    return df


def ave_evaluate(data):
    '''
    Calculate the average of each column over all cases.

    INPUT:
    data =
        label      caseID success_HS hitRate_HS success_DR hitRate_DR

        train      1AVX   0.0      0.0      0.0      0.0
        train      1AVX   1.0      1.0      1.0      1.0

        train      2ACB   0.0      0.0      0.0      0.0
        train      2ACB   1.0      1.0      1.0      1.0

        test       7CEI   0.0      0.0      0.0      0.0
        test       7CEI   1.0      1.0      1.0      1.0

        test       5ACD   0.0      0.0      0.0      0.0
        test       5ACD   1.0      1.0      1.0      1.0

    OUTPUT:
    new_data =
        label      caseID success_HS hitRate_HS success_DR hitRate_DR

        train      1AVX   0.0      0.0      0.0      0.0
        train      1AVX   1.0      1.0      1.0      1.0

        train      2ACB   0.0      0.0      0.0      0.0
        train      2ACB   1.0      1.0      1.0      1.0

        test       7CEI   0.0      0.0      0.0      0.0
        test       7CEI   1.0      1.0      1.0      1.0

        test       5ACD   0.0      0.0      0.0      0.0
        test       5ACD   1.0      1.0      1.0      1.0

    '''


    new_data = pd.DataFrame()
    for l, perf_per_case in data.groupby('label'):
        # l = 'train', 'test' or 'valid'

        # count the model number for each case
        grouped = perf_per_case.groupby('caseID')
        num_models = grouped.apply(len)
        num_cases = len(grouped)
        print(f"{l}: {num_cases} cases")

        #--
        top_N = min(num_models)
        perf_ave = pd.DataFrame()
        perf_ave['label'] = [l] * top_N

        for col in perf_per_case.columns[2:]:
            # perf_per_case.columns = ['label', 'caseID', 'success_HS', 'hitRate_HS', 'success_DR', 'hitRate_DR']
            perf_ave[col] = np.zeros(top_N)

            for _, perf_case in grouped:
                perf_ave[col] = perf_ave[col][0:top_N] + np.array(perf_case[col][0:top_N])

            perf_ave[col] = perf_ave[col]/num_cases

        new_data = pd.concat([new_data, perf_ave])

    return new_data

def evaluate(data):

    '''
    Calculate success rate and hit rate.

    <INPUT>
    data: a data frame.

           label  caseID             modelID target        DR         HS
           Test   1AVX  1AVX_ranair-it0_5286      0  0.503823   6.980802
           Test   1AVX     1AVX_ti5-itw_354w      1  0.502845 -95.158100
           Test   1AVX  1AVX_ranair-it0_6223      0  0.511688 -11.961460

    <OUTPUT>
    out_df: a data frame.
    success: binary variable, indicating whether this case is a success when evaluating its top N models.

        out_df :
             label  caseID   success_DR   hitRate_DR   success_HS   hitRate_HS
             train  1ZHI     1            0.1          0            0.01
             train  1ZHI     1            0.2          1            0.3

        where success =[0, 0, 1, 1, 1,...]: starting from rank 3 this case is a success

    '''

    out_df = pd.DataFrame()
    labels = data.label.unique() #['train', 'test', 'valid']


    for l in labels:
        # l = 'train', 'test' or 'valid'

        out_df_tmp = pd.DataFrame()

        df = data.loc[data.label == l].copy()
        methods = df.columns
        methods = methods[4:]
        df_grped = df.groupby('caseID')

        for M in methods:
#            df_sorted = df_one_case.apply(pd.DataFrame.sort_values, by= M, ascending=True)

            success = []
            hitrate = []
            caseIDs = []
            for caseID, df_one_case in df_grped:
                df_sorted = df_one_case.sort_values( by= M, ascending=True)
                hitrate.extend( rankingMetrics.hitrate(df_sorted['target'].astype(np.int)) )
                success.extend( rankingMetrics.success(rankingMetrics.hitrate(df_sorted['target'].astype(np.int)) ))
                caseIDs.extend([caseID] * len(df_one_case))

            #hitrate = df_sorted['target'].apply(rankingMetrics.hitrate) # df_sorted['target']: class IDs for each model
            #success = hitrate.apply(rankingMetrics.success) # success =[0, 0, 1, 1, 1,...]: starting from rank 3 this case is a success


            out_df_tmp['label'] = [l] * len(df) # train, valid or test
            out_df_tmp['caseID'] = caseIDs
            out_df_tmp[f'success_{M}'] = success
            out_df_tmp[f'hitRate_{M}'] = hitrate

        out_df = pd.concat([out_df, out_df_tmp])


    return out_df


def plot_evaluation(df, figname):
    '''
    INPUT:
         label   success_DR  hitRate_DR  success_HS  hitRate_HS      rank
         Test          0.0    0.000000         0.0    0.000000  0.000949
         Test          0.0    0.000000         1.0    0.012821  0.001898

         Train         0.0    0.000000         1.0    0.012821  0.002846
         Train         0.0    0.000000         1.0    0.025641  0.003795

    '''

    #---------- hit rate plot -------
    figname1 = figname + '.hitRate.png'
    print(f'\n --> Hit Rate plot:', figname1, '\n')
    hit_rate_plot(df)
    ggplot2.ggsave(figname1, height = 7 , width = 7 * 1.2, dpi = 100)


    #---------- success rate plot -------
    figname2 = figname + '.successRate.png'
    print(f'\n --> Success Rate plot:', figname2, '\n')

    success_rate_plot(df)
    ggplot2.ggsave(figname2, height = 7 , width = 7 * 1.2, dpi=100)



def hit_rate_plot(df):
    '''
    INPUT:
         label   success_DR  hitRate_DR  success_HS  hitRate_HS      rank
         Test          0.0    0.000000         0.0    0.000000  0.000949
         Test          0.0    0.000000         1.0    0.012821  0.001898

         Train         0.0    0.000000         1.0    0.012821  0.002846
         Train         0.0    0.000000         1.0    0.025641  0.003795

    '''

    #-- melt df
    df_melt = pd.melt(df, id_vars=['label', 'rank'])
    idx1 = df_melt.variable.str.contains('^hitRate')
    df_tmp = df_melt.loc[idx1,:].copy()
    df_tmp.columns = ['Sets', 'rank', 'Methods', 'hit_rate']

    tmp = list(df_tmp['Methods'])
    df_tmp.loc[:,'Methods']= [re.sub('hitRate_','',x) for x in tmp] # success_DR -> DR

    font_size = 20
    breaks = pd.to_numeric(np.arange(0,1.01,0.25))
    xlabels = list(map(lambda x: str('%d' % (x*100)) + ' % ', np.arange(0,1.01,0.25)) )
    text_style = element_text(size = font_size, family = "Tahoma", face = "bold")

    p = ggplot(df_tmp) + \
            aes_string(x='rank', y = 'hit_rate', color='Sets', linetype= 'Methods') + \
            geom_line(size=1) + \
            labs(**{'x': 'Top models (%)', 'y': 'Hit Rate'}) + \
            theme_bw() + \
            theme(**{'legend.position': 'right',
                'plot.title': text_style,
                'text': text_style,
                'axis.text.x': element_text(size = font_size),
                'axis.text.y': element_text(size = font_size)}) +\
            scale_x_continuous(**{'breaks':breaks, 'labels': xlabels})

    return p

def success_rate_plot(df):
    '''
    # INPUT: a pandas data frame
            label  success_HS  hitRate_HS  success_DR  hitRate_DR
        0  valid         1.0         1.0         0.0         0.0
        1  valid         0.0         1.0         0.0         0.0
    '''

    #-- add the 'rank' column to df
    rank = []
    for _, df_per_label in df.groupby('label'):
        num_mol = len(df_per_label)
        rank_raw = np.array(range(num_mol )) + 1
        rank.extend(rank_raw/num_mol )
    df['rank'] = rank

    #-- melt df
    df_melt = pd.melt(df, id_vars=['label', 'rank'])
    idx1 = df_melt.variable.str.contains('^success_')
    df_tmp = df_melt.loc[idx1,:].copy()
    df_tmp.columns = ['Sets', 'rank', 'Methods', 'success_rate']

    tmp = list(df_tmp['Methods'])
    df_tmp.loc[:,'Methods']= [re.sub('success_','',x) for x in tmp] # success_DR -> DR

    font_size = 20
    breaks = pd.to_numeric(np.arange(0,1.01,0.25))
    xlabels = list(map(lambda x: str('%d' % (x*100)) + ' % ', np.arange(0,1.01,0.25)) )
    text_style = element_text(size = font_size, family = "Tahoma", face = "bold")

    p = ggplot(df_tmp) + \
            aes_string(x='rank', y = 'success_rate', color='Sets', linetype= 'Methods') + \
            geom_line(size=1) + \
            labs(**{'x': 'Top models (%)', 'y': 'Success Rate'}) + \
            theme_bw() + \
            theme(**{'legend.position': 'right',
                'plot.title': text_style,
                'text': text_style,
                'axis.text.x': element_text(size = font_size),
                'axis.text.y': element_text(size = font_size)}) +\
            scale_x_continuous(**{'breaks':breaks, 'labels': xlabels})

#    p.plot()
    return p

def get_irmsd( source_hdf5, modelIDs):

    irmsd = []
    for h5FL, modelID in zip_equal(source_hdf5, modelIDs):
        # h5FL = '/home/lixue/DBs/BM5-haddock24/hdf5/000_1AY7.hdf5'
        f = h5py.File(h5FL, 'r')
        irmsd.append(f[modelID]['targets/IRMSD'][()])
        f.close()
    return irmsd



def get_HS(modelIDs,haddockS):
    HS=[]
    idx_keep = []

    for idx, modelID in enumerate(modelIDs):
        if modelID in haddockS:
            HS.append(haddockS[modelID])
            idx_keep.append(idx)
    return HS, idx_keep

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

    modelIDs = df['modelID']
    source_hdf5FLs = df['sourceFL']
    irmsd = np.array(get_irmsd(source_hdf5FLs, modelIDs))
    df['irmsd'] = irmsd
    return df



def prepare_df(deeprank_h5FL, HS_h5FL, epoch):

    '''
    OUTPUT: a data frame:

        label caseID               modelID target                                          sourceFL        DR      irmsd         HS
        Test   1AVX  1AVX_ranair-it0_5286      0  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5  0.503823  25.189108   6.980802
        Test   1AVX     1AVX_ti5-itw_354w      1  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5  0.502845   3.668682 -95.158100
    '''
    #-- read deeprank_h5FL epoch data into pd.DataFrame (DR_df)
    DR_df = read_epoch_data(deeprank_h5FL, epoch)

    '''
    DR_df (a data frame):

    label               modelID  target        DR                                          sourceFL
    0  Test  1AVX_ranair-it0_5286       0  0.503823  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    1  Test     1AVX_ti5-itw_354w       1  0.502845  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5
    2  Test  1AVX_ranair-it0_6223       0  0.511688  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5

    '''
    #-- add iRMSD column to DR_df
    DR_df = add_irmsd(DR_df)

    #-- report the number of hits for train/valid/test
    hit_statistics(DR_df)

    #-- add HS to DR_df (note: bound complexes do not have HS)
    stats = read_haddockScoreFL(HS_h5FL)
    haddockS = stats['haddock-score']# haddockS[modelID] = score
    DR_HS_df = merge_HS_DR(DR_df, haddockS)

    '''
    DR_HS_df (a data frame):

    data:
        label caseID   modelID                                          sourceFL          target   score_method1  score_method2
        train 1ZHI     1ZHI_294w  /home/lixue/DBs/BM5-haddock24/hdf5/000_1ZHI.hdf5        0       9.758          -19.3448
        test  1ACB     1ACB_89w   /home/lixue/DBs/BM5-haddock24/hdf5/000_1ACB.hdf5        1       17.535         -11.2127
    '''

    return DR_HS_df

def hit_statistics(df):

    '''
    Report the number of hits for Train, valid and test.

    INPUT (a data frame):

        label               modelID target        DR                                          sourceFL      irmsd
        Test  1AVX_ranair-it0_5286      0  0.503823  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5  25.189108
        Test     1AVX_ti5-itw_354w      1  0.502845  /home/lixue/DBs/BM5-haddock24/hdf5/000_1AVX.hdf5   3.668682
    '''

    labels = ['Train', 'Valid', 'Test']
    grouped = df.groupby('label')

    #-- 1. count num_hit based on i-rmsd
    num_hits = grouped['irmsd'].apply(lambda x: len(x[x<=4]))
    num_models = grouped.apply(len)

    for label in labels:
        print(f"According to 'i-RMSD' -> num of hits for {label}: {num_hits[label]} out of {num_models[label]} models")

    print("")
    #-- 2. count num_hit based on the 'target' column
    num_hits = grouped['target'].apply(lambda x: len(x[x=='1']))
    num_models = grouped.apply(len)

    for label in labels:
        print(f"According to 'targets' -> num of hits for {label}: {num_hits[label]} out of {num_models[label]} models")

def main(HS_h5FL= '/home/lixue/DBs/BM5-haddock24/stats/stats.h5'):
    if len(sys.argv) !=4:
        print(f"Usage: python {sys.argv[0]} epoch_data.hdf5 epoch fig_name")
        sys.exit()
    deeprank_h5FL = sys.argv[1] #the output h5 file from deeprank: 'epoch_data.hdf5'
    epoch = int(sys.argv[2]) # 9
    figname = sys.argv[3]

    pandas2ri.activate()

    df = prepare_df(deeprank_h5FL, HS_h5FL, epoch)

    #-- plot
    plot_HS_iRMSD(df, figname=figname + '.epo' + str(epoch) +'.irsmd_HS.png')
    plot_DR_iRMSD(df, figname=figname + '.epo' + str(epoch) + '.irsmd_DR.png')
    plot_boxplot(df, figname=figname + '.epo' + str(epoch) + '.boxplot.png',inverse = False)
    plot_successRate_hitRate(df[['label', 'caseID', 'modelID', 'target', 'DR','HS']].copy(), figname=figname + '.epo' + str(epoch) ,inverse = False)

if __name__ == '__main__':
    main()


