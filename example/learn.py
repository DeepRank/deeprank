import os
import glob
import numpy as np

from deeprank.learn import *
from deeprank.learn.model3d import cnn as cnn3d

database = '*1ak4.hdf5'
out = './out'

# make sure the databse is there
# for db in database:
# 	if not os.path.isfile(db):
# 	  raise FileNotFoundError('Database %s not found. Make sure to run test_generate before' %db)

# clean the output dir
if os.path.isdir(out):
  for f in glob.glob(out+'/*'):
    os.remove(f)
  os.removedirs(out)


# declare the dataset instance
data_set = DataSet(database,
            test_database = None,
            mapfly=True,
            grid_info = {'number_of_points':[30,30,30], 'resolution' : [1,1,1]},

            select_feature={'AtomicDensities' : {'CA':3.5, 'C':3.5, 'N':3.5, 'O':3.5},
                			'Features'        : ['coulomb','vdwaals','charge','PSSM_*'] },
            select_target='DOCKQ',
            tqdm=True,
            normalize_features = False,
            normalize_targets=False,
            clip_features=False,
            pair_chain_feature=np.add,
            dict_filter={'DOCKQ':'<1.'})

# create the network
model = NeuralNet(data_set,cnn3d,model_type='3d',task='reg',
                  cuda=False,plot=True,outdir=out)

# start the training
model.train(nepoch = 5,divide_trainset=0.8, train_batch_size = 5, num_workers=0, save_model='all')