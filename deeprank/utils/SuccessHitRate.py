#!/usr/bin/env python
# Li Xue
#  8-Aug-2017 19:58
"""
INPUT (*.rnk, the scoring methods have to start from the 4th column, the iRMSD column has to be called 'iRMSD')

    #modelID    model_class irmsd   HaddockScore    -iScore_KNN1
    1ZHI_294w   0           9.758   -19.3448        0.37
    1ZHI_89w    0           17.535  -11.2127        0.14

OUTPUT (hitRate.tsv and successRate.tsv):

    top HaddockScore    -iScore_KNN1
    0   0.0         0.0
    1   0.0         0.0
    2   0.25        0.0
    3   0.25        0.25


"""
import sys
import re
import os
import glob
import numpy as np
import scipy.stats as ss
import pandas as pd

def count_hits(df,col_name):
    #-- for a given target, count whether it's hits in top 1, 2, 3 ... 400 when scored based on col_name
    #
    irmsd_cutoff = 4
    num_models = df.shape[0]
    df_sorted = df.sort_values(by=col_name)

    ranks = np.floor(ss.rankdata(df_sorted[col_name]))
    df_sorted['ranks'] = ranks
    df_sorted['idx'] = range(1,num_models+1)

    print ("count hits for %s:" % col_name)
    print ("sorted df:")
    print (df_sorted)

    num_hits=np.array([-1]*(num_models +1) )
    success =np.array([-1]*(num_models +1) )
    np.put(num_hits,0,0)
    np.put(success,0,0)
    for N in np.unique(ranks):
        #print ("N=%f"%N)
        idx = df_sorted.idx[ranks==N].values.tolist()
        #print ("idx for rank = %d: %s" %(N, np.array_str(np.array(idx))) )


        df_tmp = df_sorted[ranks<= N]
        tmp = df_tmp[df_tmp.irmsd < irmsd_cutoff]
        num_hit = len(tmp)
        np.put(num_hits,idx, num_hit)
#        num_hits[idx] = [num_hit ] * len(idx)

        if num_hit >0:
            # there is at least one hit among top rank N
            np.put(success, idx, 1)
        else:
            # there is NO hit among top rank N
            np.put(success, idx, 0)

    #-- final check

    if len(success[success == -1]) != 0 or len(num_hits[num_hits== -1]) != 0:
        sys.exit("the array of success or num_hits still have some elements not defined.")

    #--
    total_numHits = num_hits[-1]
    print ("This target has a total num of hits (out of %d models): %d" % (num_models,total_numHits) )

    #-- calculate hit Rate
    if total_numHits ==0:
        hitRate = [0] * (num_models+1) #-- hitRate also has a number to top 0
    else:
        hitRate = np.divide(num_hits,total_numHits);

    a=np.array_str(np.asarray(hitRate))
    print ("HitRate for %s: %s" %(col_name, a))
    a=np.array_str(np.array(success))
    print ("Success for %s: %s" %(col_name, a))
    return (success, hitRate);

def get_successRate(success, top_ranks, methods):
    #-- calculate the success rate for 1 to top_ranks
    #-- success is a binary dictionary: success[target]


    num_methods = len(methods)
    num_targets = len(success.keys())
    #print("num_targets: %d"% num_targets)

    SuccessRate={}
    for M_idx in range(0,num_methods):
        for T in success.keys():
            # for each target, success is a 400 * num_method binary matrix,
            # indicating when returning top N models, this target has at least one hit or not
            print("count the success for target %d and method %s and top_rank %d" % (T, methods[M_idx], top_ranks))
            num_successes=np.add(num_successes,success[T][M_idx][0:top_ranks+1])
        M = methods[M_idx]
        SuccessRate[M] = np.divide(num_successes,num_targets)
    SuccessRate= pd.DataFrame(SuccessRate);

    #-- add a column for 0, 1, ..., top_ranks
    SuccessRate['top']=range(0,top_ranks+1)

    return SuccessRate



def get_hitRate(hitRate, top_ranks, methods):
    #-- calculate the average hit rate among all targets for 1 to top_ranks
    #-- hitRate is a dictionary: hitRate[target]

    num_methods = len(methods)
    num_targets = len(hitRate.keys())
    print("num_targets: %d"% num_targets)

    HitRate_final = {}
    for M_idx in range(0,num_methods):
        hitRate_sum =[0] * (top_ranks +1)
        num_successes=[0]* (top_ranks +1)
        for T in hitRate.keys():
            hitRate_sum = np.add(hitRate_sum , hitRate[T][M_idx][0:top_ranks+1])
        M=methods[M_idx]
        HitRate_final[M] = np.divide(hitRate_sum,num_targets)
    HitRate_final= pd.DataFrame(HitRate_final);

    #-- add a column for 0, 1, ..., top_ranks
    HitRate_final['top']=range(0,top_ranks+1)

    return HitRate_final

def header(fl, num_targets):
    f = open (fl, 'w')
    f.write("# generate by %s\n" % __file__)
    f.write("# total num of targets: %d\n" % num_targets)
    f.close()

def main():
    if (len(sys.argv) != 2):
        sys.exit('Usage: python SuccessRate.py dir (dir contains all target scoring files: *.rnk)')

    top_ranks = 219 #299 #-- 2G77 has only 299 water models
    dataDir = sys.argv[1]

    if not os.path.isdir(dataDir):
        sys.exit("dataDir: %s does not exist!" % dataDir)

    rnkFLs = glob.glob(dataDir + '/*.rnk')
    num_FLs = len(rnkFLs);
    if num_FLs ==0:
        sys.exit("dataDir (%s) does not contain .rnk files"% dataDir)

    print("There are %d .rnk files under %s"%(num_FLs,dataDir))
    num_targets = num_FLs;

    success={}
    hitRate={}

    for i in range(0,num_FLs):
        success[i]=[]
        hitRate[i]=[]


    #-- for each target, count num_hits and success
    for i in range(0,num_FLs):
        print ("\nRead rnk file: %s" % rnkFLs[i])
        df=pd.read_table(rnkFLs[i])
    #    print (df)

        methods = df.columns
        methods = methods[2:]

        for M in methods:
             (success_i_M, hitRate_i_M)= count_hits(df,M)
             success[i].append(success_i_M)
             hitRate[i].append(hitRate_i_M)

    #-- calculate the success rate and hit rate across all targets
    successRate = get_successRate(success,  top_ranks, methods)
    hitRate = get_hitRate(hitRate, top_ranks, methods)

#    """
    print ("SuccessRate:")
    print(successRate)

    print("\nHitRate:")
    print(hitRate)
#    """

    #-- write to file
    methods=methods.insert(0,'top')
    successRateFL=dataDir + '/successRate.tsv'
    hitRateFL=dataDir + '/hitRate.tsv'

    header(successRateFL, num_targets)
    header(hitRateFL, num_targets)
    successRate.to_csv(successRateFL, sep="\t", index=0, columns=methods, mode='a')
    hitRate.to_csv(hitRateFL, sep= "\t", index=0, columns = methods, mode='a')

    print ("%s generated" % successRateFL)
    print ("%s generated" % hitRateFL)

if __name__ == '__main__':
    main()
