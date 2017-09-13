import deeprank


##########################################################################
#
#	STEP 3 DEEP LEARNING
#
##########################################################################

import torch.optim as optim
import models3d

#adress of the database
database = './training_set/'

data_set = deeprank.DeepRankDataSet(database,
                           filter_dataset = 'decoyID.dat',
                           select_feature={'AtomicDensities' : 'all'},
                           select_target='haddock_score')

# create the network
model = deeprank.DeepRankConvNet(data_set,
                        models3d.ConvNet3D_reg,
                        model_type='3d',
                        task='reg',
                        tensorboard=False,
                        outdir='./test_out/')

# change the optimizer (optional)
model.optimizer = optim.SGD(model.net.parameters(),
                            lr=0.001,
                            momentum=0.9,
                            weight_decay=0.005)

# start the training
model.train(nepoch = 250)
