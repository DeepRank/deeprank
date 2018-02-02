import os
from deeprank.learn import *
from deeprank.learn.model3d import cnn
import torch.nn as nn

# all the import torch fails on TRAVIS
# so we can only exectute this test locally
def test_transfer():

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



  model = NeuralNet(data_set,cnn,outdir='./out_reload/')
  model.load_model('./out/model.pth.tar')

  #freeze the parameters
  for param in model.net.parameters():
    param.require_grad = False

  # get new fc layers
  size = model.net.fclayer_000.in_features
  model.net.fclayer_000   = nn.Linear(size,84)
  model.net.fclayer_001  = nn.Linear(84,1)

  model.train(nepoch = 1,percent_train=0.8, train_batch_size = 5,num_workers=0)

if __name__ == "__main__":
  test_transfer()