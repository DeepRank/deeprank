import os
from deeprank.learn import *
from deeprank.learn.model3d import cnn


# all the import torch fails on TRAVIS
# so we can only exectute this test locally
def test_transfer():

  #adress of the database
  database = '1ak4.hdf5'

  if not os.path.isfile(database):
    raise FileNotFoundError('Database %s not found. Make sure to run test_generate before')

  # create the network
  model = NeuralNet(database,cnn,pretrained_model='./out/model.pth.tar',outdir='./test/')
  model.test()

  # #freeze the parameters
  # for param in model.net.parameters():
  #   param.require_grad = False

  # # get new fc layers
  # size = model.net.fclayer_000.in_features
  # import torch.nn as nn
  # model.net.fclayer_000   = nn.Linear(size,84)
  # model.net.fclayer_001  = nn.Linear(84,1)

  # model.train(nepoch = 1,percent_train=0.8, train_batch_size = 5,num_workers=0)

if __name__ == "__main__":
  test_transfer()