import numpy as np
import pandas as pd

from deeprank.learn import rankingMetrics

def cal_hitrate_successrate(df):
    """calculate the hit rate and success_rate of the different train/valid/test
    sets with HS (haddock scores)

    The hit rate is defined as:
        the percentage of positive decoys that are included among the top m decoys.
        a positive decoy is a native-like one with a i-rmsd <= 4A

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
    """

    # -- 1. calculate success rate (SR) and hit rate (HR)
    HR_SR_per_case = evaluate(df)
    HR_SR_per_case = add_rank(HR_SR_per_case, groupby = ['label', 'caseID'])
    HR_SR_per_case = add_perc(HR_SR_per_case, groupby = ['label', 'caseID'])

    HR_SR_ave = ave_evaluate(HR_SR_per_case)
    HR_SR_ave = add_rank(HR_SR_ave, groupby = 'label')

    HR_SR_ave.label =pd.Categorical(HR_SR_ave.label, categories=["Train","Valid", "Test"])

    # -- 2. write to tsv files
    outFL1 = 'hitrate_successrate_per_case.tsv'
    HR_SR_per_case.to_csv(outFL1, index = False, sep = '\t', float_format = '%.2f')

    outFL2 = 'hitrate_successrate_ave.tsv'
    HR_SR_ave.to_csv(outFL2, index = False, sep="\t", float_format = '%.2f')

    print("")
    print (f"{outFL1} generated")
    print (f"{outFL2} generated")

    #-- 3. return value
    return HR_SR_per_case, HR_SR_ave


def evaluate(data):
    """Calculate hit rate and success rate for each case.

    <INPUT>
    data: a data frame.

           label  caseID             modelID target        DR         HS
           Test   1AVX  1AVX_ranair-it0_5286      0  0.503823   6.980802
           Test   1AVX     1AVX_ti5-itw_354w      1  0.502845 -95.158100
           Test   1AVX  1AVX_ranair-it0_6223      0  0.511688 -11.961460

    <OUTPUT>
    out_df: a data frame.
    success: binary variable, indicating whether this case is a success when evaluating its top N models.

        out_df:
             label  caseID   success_DR   hitRate_DR   success_HS   hitRate_HS
             train  1ZHI     1            0.1          0            0.01
             train  1ZHI     1            0.2          1            0.3

        where success =[0, 0, 1, 1, 1,...] means: starting from rank 3 this case is a success
    """

    out_df = pd.DataFrame()
    labels = data.label.unique()  # ['train', 'test', 'valid']

    for l in labels:
        # l = 'train', 'test' or 'valid'

        out_df_tmp = pd.DataFrame()

        df = data.loc[data.label == l].copy()
        methods = df.columns
        methods = methods[4:] # ['DR', 'HS']
        df_grped = df.groupby('caseID')

        for M in methods:
            #            df_sorted = df_one_case.apply(pd.DataFrame.sort_values, by= M, ascending=True)

            success = []
            hitrate = []
            caseIDs = []
            for caseID, df_one_case in df_grped:
                df_sorted = df_one_case.sort_values(by=M, ascending=True)
                hitrate.extend(rankingMetrics.hitrate(
                    df_sorted['target'].astype(np.int)))
                success.extend(rankingMetrics.success(
                    df_sorted['target'].astype(np.int)))
                caseIDs.extend([caseID] * len(df_one_case))

            # success =[0, 0, 1, 1, 1,...]: starting from rank 3 this case is a success

            out_df_tmp['label'] = [l] * len(df)  # train, valid or test
            out_df_tmp['caseID'] = caseIDs
            out_df_tmp[f'success_{M}'] = success
            out_df_tmp[f'hitRate_{M}'] = hitrate

        out_df = pd.concat([out_df, out_df_tmp])
    out_df.label = pd.Categorical(out_df.label, categories=['Train', 'Valid', 'Test'])

    return out_df

def ave_evaluate(data):
    """Calculate the average of each column over all cases.

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
         label  successRate_DR  hitRate_DR  successRate_HS  hitRate_HS
         Test         0.0         0.0         0.0         0.0
         Test         0.0         0.0         0.0         0.0
         Test         0.2         0.1         0.3         0.0
         Test         1.0         0.4         1.0         0.3

        Train         1.0         0.1         1.0         0.1
        Train         1.0         0.3         1.0         0.2
        Train         1.0         0.4         1.0         0.3
        Train         1.0         0.9         1.0         0.5

        Valid         0.0         0.2         1.0         0.1
        Valid         1.0         0.4         1.0         0.3
        Valid         1.0         0.7         1.0         0.5
        Valid         1.0         0.9         1.0         0.7

    """

    num_cases, num_models = count(data)

    new_data = pd.DataFrame()
    for l, perf in data.groupby('label'):
        # l = 'Train', 'Test' or 'Valid'

        top_N = min(num_models[l])
        print(f"Calculate hitrate/successrate over {num_cases[l]} cases in {l} on top 1-{top_N} models.")

        perf_ave = pd.DataFrame()
        perf_ave['label'] = [l] * top_N

        for col in perf.columns[perf.columns.str.contains('^(hitRate_|success|perc)', case= False)]:
            # col = 'success_HS', 'hitRate_HS', 'success_DR', 'hitRate_DR'
            perf_ave[col] = np.zeros(top_N)

            for _, perf_case in perf.groupby('caseID'):
                perf_case = perf_case.reset_index()
                perf_ave[col] = perf_ave[col] + \
                    np.array(perf_case.loc[0:top_N-1, col])

            perf_ave[col] = perf_ave[col] / num_cases[l]

        new_data = pd.concat([new_data, perf_ave])

    return new_data

def count(df):
    '''Count the number of cases and the number of models per case

    INPUT:
    df =
        label      caseID success_HS hitRate_HS success_DR hitRate_DR

        train      1AVX   0.0      0.0      0.0      0.0
        train      1AVX   1.0      1.0      1.0      1.0

    OUTPUT (a pd series):

    num_cases =
        label
        Train    114
        Valid     14
        Test      14

    num_models =
        label   caseID
        Train   1AK4      1054
                1ATN       870

    '''

    grp = df.groupby(['label', 'caseID'])
    num_models = grp.apply(len)

    grp = num_models.groupby(['label'])
    num_cases = grp.apply(len)

    return num_cases, num_models

def add_rank(df, groupby = ['label', 'caseID']):
    """
    groupby (list or str): add rank by the specified column(s) -> ['label', 'caseID']

    INPUT (a data frame):
        label   success_DR  hitRate_DR  success_HS  hitRate_HS
        Test          0.0    0.000000         0.0    0.000000
        Test          0.0    0.000000         1.0    0.012821

        Train         0.0    0.000000         1.0    0.012821
        Train         0.0    0.000000         1.0    0.025641

    OUTPUT:
         label   success_DR  hitRate_DR  success_HS  hitRate_HS      rank
         Test          0.0    0.000000         0.0    0.000000       1
         Test          0.0    0.000000         1.0    0.012821       2

         Train         0.0    0.000000         1.0    0.012821       1
         Train         0.0    0.000000         1.0    0.025641       2
    """

    # -- add the 'rank' column to df
    frames = [] # dfs for train/valid/test, respectively
    rank = []
    for _, df_per_grp in df.groupby(groupby):
        num_mol = len(df_per_grp)
        rank_raw = np.array(range(num_mol)) + 1
        tmp_df = df_per_grp.copy()
        tmp_df['rank'] = rank_raw
        frames.append(tmp_df)

    new_df = pd.concat(frames)

    return new_df

def add_perc(df, groupby = ['label', 'caseID']):
    """
    groupby (list or str): add perc by the specified column(s) -> ['label', 'caseID']

    INPUT (a data frame):
        label   success_DR  hitRate_DR  success_HS  hitRate_HS
        Test          0.0    0.000000         0.0    0.000000
        Test          0.0    0.000000         1.0    0.012821

        Train         0.0    0.000000         1.0    0.012821
        Train         0.0    0.000000         1.0    0.025641

    OUTPUT:
         label   success_DR  hitRate_DR  success_HS  hitRate_HS      perc
         Test          0.0    0.000000         0.0    0.000000  0.000949
         Test          0.0    0.000000         1.0    0.012821  0.001898

         Train         0.0    0.000000         1.0    0.012821  0.002846
         Train         0.0    0.000000         1.0    0.025641  0.003795
    """

    # -- add the 'perc' column to df
    frames = [] # dfs for train/valid/test, respectively
    rank = []
    for _, df_per_grp in df.groupby(groupby):
        num_mol = len(df_per_grp)
        rank_raw = np.array(range(num_mol)) + 1
        tmp_df = df_per_grp.copy()
        tmp_df['perc'] = rank_raw/num_mol
        frames.append(tmp_df)

    new_df = pd.concat(frames)

    return new_df
