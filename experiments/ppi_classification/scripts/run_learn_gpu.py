"""
Training 3DeepFace models

Usage: python {2} <Output folder name> <batch size> <number of workers> <use number of rotations>
Example: python {2} out_model_test 2 2 0

Author: {0} ({1})
"""
import os
import sys
import glob
from deeprank.learn import *

# to set your own architecture
# arch_001: input 40channels 30x30x30
# arch_002: input 2channels 30x30x30
# arch_003: input 42channels 30x30x30
#  from deeprank.learn.arch_001 import cnn_class as cnn3d_class
from arch_001_02 import cnn_class as cnn3d_class
#  from deeprank.learn.arch_002 import cnn_class as cnn3d_class
#  from deeprank.learn.arch_003 import cnn_class as cnn3d_class


__author__ = "Cunliang Geng"
__email__ = "gengcunliang AT gmail.com"
USAGE = __doc__.format(__author__, __email__, __file__)


def check_input(args):
    if len(args) != 4:
        sys.exit(USAGE)


################################################################################
# input and output settings
################################################################################

trainset = ['/projects/0/deepface/MANY/hdf5/train_bio.hdf5',
            '/projects/0/deepface/MANY/hdf5/train_xtal.hdf5']

validset = ['/projects/0/deepface/MANY/hdf5/valid_bio.hdf5',
            '/projects/0/deepface/MANY/hdf5/valid_xtal.hdf5']

testdb = ['/projects/0/deepface/DC/hdf5/dcbio_01.hdf5',
          '/projects/0/deepface/DC/hdf5/dcxtal_01.hdf5']

# to set output folder
outpath = '/projects/0/deepface/training'

check_input(sys.argv[1:])

outdir = sys.argv[1]
out = os.path.join(outpath, outdir)
batchsize = int(sys.argv[2])
workers = int(sys.argv[3])
use_rotation = int(sys.argv[4])


################################################################################
# Start the training
################################################################################

# remove output folder if exist
if os.path.isdir(out):
    for f in glob.glob(out+'/*'):
        os.remove(f)
else:
    os.mkdir(out)

# declare the dataset instance
data_set = DataSet(train_database=trainset,
                   valid_database=validset,
                   test_database=testdb,
                   mapfly=False,  # Features have been already mapped
                   # mapfly not properly tested yet
                   # grid_info = {'number_of_points' : [30,30,30],
                   # 'resolution' : [1,1,1]}, # Grid has been already calculated
                   use_rotation=use_rotation,
                   # select_feature={'Feature_ind': ['AtomicFeature', 'BSA', 'ResidueDensity']},
                   select_feature={'Feature_ind' : ['PSSM_*'] }, ## 40 features, to use arch_001
                   # select_feature={'Feature_ind' : ['pssm_ic_*']}, ## 2 features, to use arch_002
                   # select_feature={'Feature_ind' : ['PSSM_*', 'pssm_ic_*'] },
                   # select_feature='all',
                   select_target='BIN_CLASS',
                   normalize_features=False,
                   normalize_targets=False,
                   pair_chain_feature=None,
                   clip_features=False,
                   tqdm=True,
                   process=True)  # process must be False for test


# create the network
model = NeuralNet(data_set=data_set,
                  model=cnn3d_class,
                  model_type='3d',
                  task='class',
                  pretrained_model=None,
                  cuda=True,
                  ngpu=2,
                  plot=True,
                  save_hitrate=False,
                  save_classmetrics=True,
                  outdir=out)


# start the training
model.train(nepoch=30,
            preshuffle=True,
            preshuffle_seed=2019,
            divide_trainset=None,
            train_batch_size=batchsize,
            num_workers=workers,
            save_model='all',
            #  save_model='best',
            save_epoch='all',
            hdf5='epoch_data.hdf5'
            )
