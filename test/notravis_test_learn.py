import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from deeprank.learn import *
from deeprank.learn.model3d import cnn


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
                            select_feature = 'all',select_target='DOCKQ',tqdm=True,
                            normalize_features = True, normalize_targets=True)


  # create the network
  model = NeuralNet(data_set,cnn,model_type='3d',task='reg',
                    cuda=False,plot=True,outdir='./out/')

  # start the training
  model.train(nepoch = 50,percent_train=0.8, train_batch_size = 5,num_workers=0)

  # save the model
  #model.save_model()

  # reload the model and test it
  # declare the dataset instance
  #data_set = InMemoryDataSet(database,test_database = database,
  #                          select_feature = 'all',select_target='DOCKQ',tqdm=True,
  #                          normalize_features = False, normalize_targets=False)

  #model = NeuralNet(data_set,ConvNet3D,outdir='./out_reload/')
  #model.load_model('./out/model.pth.tar')
  #model.data_set.normalize_data()
  #model.test()

if __name__ == "__main__":
  test_learn()
