import numpy as np
import pandas as pd

from deeprank.learn import rankingMetrics


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
    labels = data.label.unique()  # ['train', 'test', 'valid']

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
                df_sorted = df_one_case.sort_values(by=M, ascending=True)
                hitrate.extend(rankingMetrics.hitrate(
                    df_sorted['target'].astype(np.int)))
                success.extend(rankingMetrics.success(
                    df_sorted['target'].astype(np.int)))
                caseIDs.extend([caseID] * len(df_one_case))

            # hitrate = df_sorted['target'].apply(rankingMetrics.hitrate) # df_sorted['target']: class IDs for each model
            # success = hitrate.apply(rankingMetrics.success) # success =[0, 0, 1, 1, 1,...]: starting from rank 3 this case is a success

            out_df_tmp['label'] = [l] * len(df)  # train, valid or test
            out_df_tmp['caseID'] = caseIDs
            out_df_tmp[f'success_{M}'] = success
            out_df_tmp[f'hitRate_{M}'] = hitrate

        out_df = pd.concat([out_df, out_df_tmp])

    return out_df


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

        # --
        top_N = min(num_models)
        perf_ave = pd.DataFrame()
        perf_ave['label'] = [l] * top_N

        for col in perf_per_case.columns[2:]:
            # perf_per_case.columns = ['label', 'caseID', 'success_HS', 'hitRate_HS', 'success_DR', 'hitRate_DR']
            perf_ave[col] = np.zeros(top_N)

            for _, perf_case in grouped:
                perf_ave[col] = perf_ave[col][0:top_N] + \
                    np.array(perf_case[col][0:top_N])

            perf_ave[col] = perf_ave[col]/num_cases

        new_data = pd.concat([new_data, perf_ave])

    return new_data


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

    # -- add the 'rank' column to df
    rank = []
    for _, df_per_label in df.groupby('label'):
        num_mol = len(df_per_label)
        rank_raw = np.array(range(num_mol)) + 1
        rank.extend(rank_raw/num_mol)
    df['rank'] = rank

    df['label'] = pd.Categorical(df['label'], categories=[
                                 'Train', 'Valid', 'Test'])

    return df
