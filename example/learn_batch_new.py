import glob
import os
import pdb
import re
import sys
from math import *

import numpy as np

from deeprank.learn import *
from model3d import cnn_class as cnn3d
from torch import optim

"""
An example to do cross-validation 3d_cnn at the case level
(i.e., all docked models of one case will belong either to training, valiation or test only)
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def divide_data(hdf5_DIR, caseID_FL, portion=[0.8,0.1,0.1], random =True, write_to_file = True):
    # INPUT: the dir that stores all hdf5 data (training, validation, and test)
    # OUPUT: randomly divide them into train, validation, and test at the caseID-level. Return the filenames.
    # write_to_file: True then write the files of trainSet.txt, valiatonSet.txt and testSet.txt

    if sum(portion) > 1:
        sys.exit("Error: The sum of portions for train/validatoin/test is larger than 1!")

    if len(portion) != 3:
        sys.exit("Error: the length of portions has to be 3.")


    caseIDs = np.array(read_listFL(caseID_FL))
    train_caseIDs, valid_caseIDs, test_caseIDs = random_split(caseIDs, portion, random = random)

    print (f"\nnum of training cases: {len(train_caseIDs)}")
    print (f"num of validation cases: {len(valid_caseIDs)}")
    print (f"num of test cases: {len(test_caseIDs)}\n")

    train_database = get_hdf5FLs(train_caseIDs, hdf5_DIR)
    valid_database = get_hdf5FLs(valid_caseIDs, hdf5_DIR)
    test_database = get_hdf5FLs(test_caseIDs, hdf5_DIR)

    print (f"\nnum of training hdf5 files: {len(train_database)}")
    print (f"num of validation hdf5 files: {len(valid_database)}")
    print (f"num of test hdf5 files: {len(test_database)}\n")

    if write_to_file is True:
        #outDIR = hdf5_DIR
        outDIR = os.getcwd()
        write_train_valid_testFLs (train_database, valid_database, test_database, outDIR)
    return train_database, valid_database, test_database

def get_hdf5FLs(caseIDs, hdf5_DIR):

    hdf5_FLs = []
    for caseID in caseIDs:
        hdf5_FLs.extend(glob.glob(f"{hdf5_DIR}/000*{caseID}.hdf5"))

    return hdf5_FLs

def read_listFL(listFL):

    f = open(listFL,'r')
    caseIDs = f.readlines()
    f.close()

    caseIDs = [ x.strip() for x in caseIDs if not re.search('^#', x) and not re.search('^\s*$',x) ]

    print (f"{len(caseIDs)} cases read from {listFL}")
    return caseIDs


def random_split(array, portion, random = True):
    # array: np.array. Can be a list of caseIDs or a list of hdf5 file names

    if random is False:
        np.random.seed(999)
    np.random.shuffle(array)

    n_cases = len(array)
    n_train = min(ceil(n_cases * portion[0]), n_cases)
    n_valid = floor(n_cases * portion[1])

    if sum(portion) == 1:
        n_test = n_cases - n_train - n_valid
    else:
        n_test = floor(n_cases * portion[2])

    train = array[:n_train]
    valid = array[n_train:n_train+n_valid]
    test  = array[n_train + n_valid: n_train + n_valid + n_test]

    return train, valid, test


def write_train_valid_testFLs (train_database, valid_database, test_database, outDIR):
    trainID_FL = f"{outDIR}/trainIDs.txt"
    validID_FL = f"{outDIR}/validIDs.txt"
    testID_FL = f"{outDIR}/testIDs.txt"

    outFLs = [trainID_FL, validID_FL, testID_FL]
    databases = [train_database, valid_database, test_database]

    for outFL, database in zip(outFLs, databases):

        if database is not True:
            np.savetxt(outFL, database, delimiter = "\n", fmt = "%s")
            print(f"{outFL} generated.")


def main():

    hdf5_DIR = '/projects/0/deeprank/BM5/hdf5' # stores all *.hdf5 files
    caseID_FL = '/projects/0/deeprank/BM5/caseID_dimers.lst'
#    hdf5_DIR = '/projects/0/deeprank/BM5/hdf5'
#    caseID_FL = '/projects/0/deeprank/BM5/caseID_dimers.lst'
    train_database, valid_database, test_database = \
        divide_data(hdf5_DIR = hdf5_DIR,caseID_FL = caseID_FL, portion = [0.6,0.1,0.1], random = False)

    # clean the output dir
    out = './out'
    if os.path.isdir(out):
        for f in glob.glob(out+'/*'):
            os.remove(f)
        os.removedirs(out)



    # declare the dataset instance

    data_set = DataSet(train_database = train_database,
                valid_database = valid_database,
                test_database = test_database,
                mapfly=False,
                use_rotation=0,
                grid_info = {'number_of_points':[6, 6, 6], 'resolution' : [5,5,5]},

                #            select_feature={'AtomicDensities' : {'CA':1.7, 'C':1.7, 'N':1.55, 'O':1.52},
    #                			'Features'        : ['coulomb','vdwaals','charge','PSSM_*'] },
                #select_feature = 'all',
                select_feature = {'Feature_ind':['coulomb']},
                select_target='BIN_CLASS',
                tqdm=True,
                normalize_features = False,
                normalize_targets=False,
                clip_features=False,
                pair_chain_feature=np.add,
                dict_filter={'DOCKQ':'>0.01', 'IRMSD':'<=4 or >10'})

    # create the networkt
    model = NeuralNet(data_set=data_set,
                    model=cnn3d,
                    model_type='3d',
                    task='class',
                    pretrained_model=None,
                    cuda=True,
                    ngpu=1,
                    plot=True,
                    save_hitrate=True,
                    save_classmetrics=True,
                    outdir=out)



    # change the optimizer (optional)
    model.optimizer = optim.SGD(model.net.parameters(),
                    lr=0.0001,momentum=0.9,weight_decay=0.0001)

    # start the training
    model.train(nepoch=1,
                preshuffle = True,
                preshuffle_seed = 2019,
                divide_trainset=None,
                train_batch_size=10,
                num_workers=6,
                save_model='all',
                save_epoch='all',
                hdf5='xue_epoch_data.hdf5'
                )

if __name__ == '__main__':
    main()
