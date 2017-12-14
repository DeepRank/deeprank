from deeprank.learn import *
import model3d
import torch.optim as optim

# declare the dataset instance
database = './1ak4.hdf5'
data_set = DataSet(database,
                  select_feature={'AtomicDensities_diff' : ['C','CA','O','N'], 
                                  'atomicFeature_sum' : ['coulomb','vdwaals','charge'] },
                  select_target='DOCKQ')


# load the data set
data_set.load()

# create the network
model = ConvNet(data_set, model3d.cnn,cuda=False)

# change the optimizer (optional)
model.optimizer = optim.SGD(model.net.parameters(),
                            lr=0.001,
                            momentum=0.9,
                            weight_decay=0.005)

# start the training
model.train(nepoch = 250)