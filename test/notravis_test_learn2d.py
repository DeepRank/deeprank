import os
import numpy as np
from deeprank.learn import *
from deeprank.learn.model2d import cnn


# all the import torch fails on TRAVIS
# so we can only exectute this test locally
def test_learn():

  #adress of the database
  database = '1ak4.hdf5'

  if not os.path.isfile(database):
    raise FileNotFoundError('Database %s not found. Make sure to run test_generate before')

  # declare the dataset instance
  data_set = DataSet(database,
            test_database = database,
            select_feature={'AtomicDensities_ind' : 'all',
                            'Feature_ind' : ['coulomb','vdwaals','charge'] },
            pair_chain_feature=None,
            select_target='DOCKQ')


  # create the network
  model = NeuralNet(data_set,cnn,model_type='2d',task='reg',
                    cuda=False,plot=True,outdir='./out/')

  # start the training
  model.train(nepoch = 50,divide_trainset=0.8, train_batch_size = 5)

  # save the model
  model.save_model()

if __name__ == "__main__":
  test_learn()
