import torch
from deeprank.learn import NeuralNet
from model_280619 import cnn_class
import glob
model_data = '.best_train_model.pth.tar'

### Change the path to point to your own graphs
database = glob.glob('./test/*.hdf5')

model = NeuralNet(database, cnn_class,
                  pretrained_model=model_data, save_hitrate=False)
model.test()
