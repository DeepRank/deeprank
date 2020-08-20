import os
import glob
import numpy as np
from deeprank.learn import *
from math import *
import sys
import re
from model_280619 import cnn_class as cnn3d
from torch import optim
import torch
import pdb

#training with class weights for balancing the error, not filtering out the data
"""
An example to do cross-validation 3d_cnn at the case level
(i.e., all docked models of one case will belong either to training, valiation or test only)
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def divide_data(hdf5_DIR, caseID_FL, portion=[0.8,0.1,0.1], n=0, random =True, write_to_file = True):
    # INPUT: the dir that stores all hdf5 data (training, validation, and test)
    # OUTPUT: divide them into train, validation, and test at the caseID-level. Return the filenames.
    # write_to_file: True then write the files of trainSet.txt, valiatonSet.txt and testSet.txt

    if sum(portion) > 1:
        sys.exit("Error: The sum of portions for train/validatoin/test is larger than 1!")

    if len(portion) != 3:
        sys.exit("Error: the length of portions has to be 3.")


    caseIDs = np.array(read_listFL(caseID_FL))
    train_caseIDs, valid_caseIDs, test_caseIDs = n_split(caseIDs, n, random = random)

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
        hdf5_FLs.extend(glob.glob(f"{hdf5_DIR}/001*{caseID}.hdf5"))
        hdf5_FLs.extend(glob.glob(f"{hdf5_DIR}/002*{caseID}.hdf5"))
        hdf5_FLs.extend(glob.glob(f"{hdf5_DIR}/003*{caseID}.hdf5"))
   
    return hdf5_FLs

def read_listFL(listFL):

    f = open(listFL,'r')
    caseIDs = f.readlines()
    f.close()

    caseIDs = [ x.strip() for x in caseIDs if not re.search('^#', x) and not re.search('^\s*$',x) ]

    print (f"{len(caseIDs)} cases read from {listFL}")
    return caseIDs




def n_split(array, n, random = True):#defines the splitting of data for the fold n
    # array: np.array. Can be a list of caseIDs or a list of hdf5 file names

    if random is False:
        np.random.seed(999)
    np.random.shuffle(array)

    n_cases = len(array)
    n_test = floor(n_cases/10) # 10% of the cases for testing 
    n_valid = floor(n_cases/10) # 10% of the cases for validation
    n_train = n_cases - n_test - n_valid # the rest for training
    n = int(n) 
    test  = array[(n-1)*n_test: n*n_test] # the set of test cases will iterate through the 10 loops
    before_test = array[:(n-1)*n_test] # the cases before the test cases
    after_test = array[n*n_test:] # the cases after the test cases
    train_valid = np.concatenate((before_test, after_test)) # concatenate to obtain the train/validation set for this loop
    start_val = (n-1)%9 #this will ensure that also the validation set is different in every loop
    valid = train_valid[start_val*n_valid: (start_val+1)*n_valid] # the val set is immediately after the test set, in round robin manner
    before_valid = train_valid[:start_val*n_valid]
    after_valid = train_valid[(start_val+1)*n_valid:]
    train = np.concatenate((before_valid, after_valid))
    
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
    n = sys.argv[1]
    hdf5_DIR = '/projects/0/deeprank/BM5/hdf5_premap_prealign_rotated' # stores all *.hdf5 files
    caseID_FL = '/projects/0/deeprank/BM5/caseID_dimers.lst'

    train_database, valid_database, test_database = \
        divide_data(hdf5_DIR = hdf5_DIR,caseID_FL = caseID_FL,  n=n, random = False)

    # clean the output dir
    out = './out_001_3_2806_prealigned_rotated_n_' + str(n)
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
                #grid_info = {'number_of_points':[6, 6, 6], 'resolution' : [5,5,5]},

                #            select_feature={'AtomicDensities' : {'CA':1.7, 'C':1.7, 'N':1.55, 'O':1.52},
    #                			'Features'        : ['coulomb','vdwaals','charge','PSSM_*'] },
                select_feature = 'all',
                #select_feature = {'Feature_ind':['coulomb']},
                select_target='BIN_CLASS',
                tqdm=True,
                normalize_features = False,
                normalize_targets=False,
                clip_features=False,
                pair_chain_feature=np.add)
             
    
    weights = [1.0/(3343981-237480), 1.0/237480]
    class_weights = torch.FloatTensor(weights).cuda()
    # create the network
    model = NeuralNet(data_set=data_set,
                    model=cnn3d,
                    class_weights = class_weights,  
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
    model.optimizer = optim.Adam(params = model.net.parameters())

    # start the training
    model.train(nepoch=1,
                preshuffle = True,
                preshuffle_seed = 2019,
                divide_trainset=None,
                train_batch_size=100,
                num_workers=15,
                save_model='all',
                save_epoch='all',
                hdf5='epoch_data.hdf5'
                )

if __name__ == '__main__':
    main()
