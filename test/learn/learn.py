import deeprank.learn
import torch.optim as optim
import torch
import models3d
import sys 
from time import time

#adress of the database
database = '../generate/1ak4.hdf5'

# declare the dataset instance
data_set = deeprank.learn.DataSet(database,
                          select_feature={'AtomicDensities_sum' : ['C','CA','O','N'], 
                                          'atomicFeature_sum' : ['coulomb','vdwaals','charge'] },
                          select_target='DOCKQ',normalize_features=True,normalize_targets=True)


# load the data set
data_set.load()

# create the network
model = deeprank.learn.ConvNet(data_set,
                        models3d.ConvNet3D,
                        model_type='3d',
                        task='reg',
                        tensorboard=False,
                        cuda=False,
                        plot=False,
                        outdir='./out/')

# start the training
model.train(nepoch = 5,divide_set=[0.8,0.1,0.1], train_batch_size = 5)
