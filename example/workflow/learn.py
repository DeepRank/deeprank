import deeprank.learn
import torch.optim as optim
import models3d
import sys 

##########################################################################
#
#	STEP 3 DEEP LEARNING
#
##########################################################################


#adress of the database
database = '../../database/'

# declare the dataset instance
#data_set = deeprank.learn.DeepRankDataSet(database,
#                           filter_dataset = 'decoyID.dat',
#                           select_feature={'AtomicDensities_diff' : ['CD','CE','O'],
#                                           'atomicFeature' : ['ELEC','VDW']  },
#                           select_target='haddock_score')
data_set = deeprank.learn.DeepRankDataSet(database,
                           select_feature={'AtomicDensities_diff' : ['CD','CE','O'],
                                           'atomicFeature' : ['ELEC','VDW']  },
                           select_target='binary_class')

# Get the content of the dataset
#data_set.get_content()

# load the data set
data_set.load()

# create the network
model = deeprank.learn.DeepRankConvNet(data_set,
                        models3d.ConvNet3D_binclass,
                        model_type='3d',
                        task='class',
                        tensorboard=False,
                        cuda=False,
                        outdir='./test_class/')


# change the optimizer (optional)
model.optimizer = optim.SGD(model.net.parameters(),
                            lr=0.001,
                            momentum=0.9,
                            weight_decay=0.005)

# start the training
model.train(nepoch = 250)
