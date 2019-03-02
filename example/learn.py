import os
import glob
import numpy as np

from deeprank.learn import *
from deeprank.learn.model3d import cnn as cnn3d

#adress of the database
train_database = './data/1AVX.hdf5'
valid_database = './data/1BVN.hdf5'
test_database = './data/1DFJ.hdf5'


# make sure the databse is there

for database in [train_database, valid_database, test_database]:
    if not os.path.isfile(database):
        raise FileNotFoundError('Database %s not found. Make sure to run test_generate before')

# clean the output dir
out = './out_3d'
if os.path.isdir(out):
  for f in glob.glob(out+'/*'):
    os.remove(f)
  os.removedirs(out)


# declare the dataset instance
data_set = DataSet(train_database = train_database,
            valid_database=valid_database,
            test_database = test_database,
            grid_shape=(30,30,30),
            select_feature={'AtomicDensities_ind' : 'all',
                            'Feature_ind' : ['coulomb','vdwaals','charge','PSSM_*'] },
            select_target='DOCKQ',
            tqdm=True,
            normalize_features = True,
            normalize_targets=True,
            clip_features=False,
            pair_chain_feature=np.add,
            dict_filter={'DOCKQ':'<1.'})

# create the network
model = NeuralNet(data_set,cnn3d,model_type='3d',task='reg',
                  cuda=False,plot=True,outdir=out)

# start the training
model.train(nepoch = 5, divide_trainset = None, train_batch_size = 5, num_workers=0, save_model='all')
