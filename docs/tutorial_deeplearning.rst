Tutorial : Deep learning
=========================

This page gives a introduction of thedeep learning process in DeepRank. This is a breakdown of the file ``test/notravis_test_learn.py``.

The deep learning module of DeepRank allows to use the data stored in the  HDF5 files in pyTorch and run deep learning experiments using different combinations of conformations, features, targets, network architecture. We will illutrate how the process work

Creating the dataset
----------------------------

The two first lines of the code are to import the module we need

>>> from deeprank.learn import *
>>> from deeprank.learn.model3d import cnn

The first lines import the ``DataSet`` and ``NeuralNetwork`` class that are in charge of deep learning. The second import a pre-generated 3D convolution neural network. This file has been generated automatically using ``deeprank.learn.modelGenerator``.

The first thing we then need to do is to define which database contains the information we want to use to train the network. This is done by:

>>> database = '1ak4.hdf5'

This is the file we have generated above. Note that more than one file can be specified here. Hence the command

>>> database = ['1ak4.hdf5','1atn.hdf5']

will use all the data contained in both of these files to train the network. We can now create an instance of deeprank.DataSet

>>> data_set = DataSet(database,
>>>                    grid_info={
>>>                    'number_of_points': (10, 10, 10),
>>>                    'resolution': (3, 3, 3)},
>>>                    select_feature={
>>>                     'AtomicDensities_ind' : 'all',
>>>                     'Feature_ind' : ['coulomb','vdwaals','charge','pssm']},
>>>                    pair_chain_feature=np.add,
>>>                    select_target='IRMSD',
>>>                    dict_filter={'IRMSD':'<4. or >10.'})

On top of the database that specify which files to use we also must specify which features and which targets must be used during the traing. The feature selection is done via the argument ``select_features``. Several options are possible for this argument.

The simplest (and default) option is to use all the features stored in the HDF5 file. To do that simply use:

>>> # select all the features
>>> select_feature = 'all'

Most of the time one want to specify which feature to use. We must then use a *dictionary*. As you cans ee in the HDF5 file, the ``mapped_features`` subgroup of each complex contains two groups ``AtomicDensities_ind`` and ``Feature_ind``. We'll forget about the ``_ind`` that is for legacy reasons, but these two groups contains the atomic densities and other features respectively. Therefore the value used in the example above:

>>> select_feature={ 'AtomicDensities_ind' : 'all',
>>>                  'Feature_ind' : ['coulomb','vdwaals','charge','pssm']}

Specify that we want to use all the atomic densities and only a few features that are named here. If for example one want to only use the pssm_ic feature then the arguments should be set to:

>>> select_feature={'Feature_ind':['pssm_ic']}

As you can see in the HDF5 file the for each feature, the data of chainA and chainB are stored separately. But it is possible to combine the feature of both chains in a single channel via the argument ``pair_chain_feature``. For example here this argument is set to:

>>> pair_chain_feature = np.add

which means that the map of chainA and chainB will be added to each other to create a single channel. By default this ``pair_chain_feature=None`` and therefore the individual maps are kept.

You must also specify which target values must be used for the training. It is the IRMSD of the complex. Finally it is possible to screen the complex contained in ``database`` and only select a few via the ``dict_filter`` argument. For example here

>>> dict_filter={'IRMSD':'<4. or >10.'}

will only select the complexes whose IRMSD are inferior to 4. and superior to 10. Angs. If one wants to select the complexes with only high dockQ score, one can use

>>> dict_filter={'DOCKQ':'>0.2'}

Other filter can be set similarly.


Creating the Neural Network
-----------------------------

DeepRank excpets the definition of the neural network to be given in a rather well defined format. Example of 2D and 3D CNN are given in ``learn/model2d.py`` and ``learn/model3d.py``. Here is for example the defintion of a simple 3D CNN

>>> import torch
>>> from torch.autograd import Variable
>>> import torch.nn as nn
>>> import torch.nn.functional as F
>>>
>>> class cnn(nn.Module):
>>>
>>>     def __init__(self,input_shape):
>>>         super(cnn,self).__init__()
>>>
>>>         self.convlayer_000 = nn.Conv3d(input_shape[0],4,kernel_size=2)
>>>         self.convlayer_001 = nn.MaxPool3d((2,2,2))
>>>         self.convlayer_002 = nn.Conv3d(4,5,kernel_size=2)
>>>         self.convlayer_003 = nn.MaxPool3d((2,2,2))
>>>
>>>         size = self._get_conv_output(input_shape)
>>>
>>>         self.fclayer_000 = nn.Linear(size,84)
>>>         self.fclayer_001 = nn.Linear(84,1)
>>>
>>>
>>>     def _get_conv_output(self,shape):
>>>         inp = Variable(torch.rand(1,*shape))
>>>         out = self._forward_features(inp)
>>>         return out.data.view(1,-1).size(1)
>>>
>>>     def _forward_features(self,x):
>>>         x = F.relu(self.convlayer_000(x))
>>>         x = self.convlayer_001(x)
>>>         x = F.relu(self.convlayer_002(x))
>>>         x = self.convlayer_003(x)
>>>         return x
>>>
>>>     def forward(self,x):
>>>         x = self._forward_features(x)
>>>         x = x.view(x.size(0),-1)
>>>         x = F.relu(self.fclayer_000(x))
>>>         x = self.fclayer_001(x)
>>>         return x

In the ``__init__`` all the convolution and fully conected layers are defined. We can here specfiy the kernel size, strides, input/output size of each layer. method ``_get_conv_output()`` allows to automatically determine the input size of the first fully connected layer. This  method relies on the ``_forward_features()`` method that passes the input data through the convolutional stage. Finally the ``forward`` method is required by pyTorch to use the network.

To facilitate the creation of these files, an automatic generator has been developped. This is the class ``modelGenerator`` that is defined in the file ``learn/modelGenerator.py``. For example the creation of the file above can be done with the following code :


>>> from deeprank.learn.modelGenerator import *
>>>
>>> conv_layers = []
>>> conv_layers.append(conv(output_size=4,kernel_size=2,post='relu'))
>>> conv_layers.append(pool(kernel_size=2))
>>> conv_layers.append(conv(input_size=4,output_size=5,kernel_size=2,post='relu'))
>>> conv_layers.append(pool(kernel_size=2))
>>>
>>> fc_layers = []
>>> fc_layers.append(fc(output_size=84,post='relu'))
>>> fc_layers.append(fc(input_size=84,output_size=1))
>>>
>>> gen = NetworkGenerator(name='test',fname='model_test.py',
>>>                      conv_layers=conv_layers,fc_layers=fc_layers)
>>> gen.print()
>>> gen.write()

As you can see all you have to do is to create to list of neural netwok layers, one for the convolutional stage and the other for the fully connected stage. Then simply feed that to the generator and write the model to file !

The classes ``conv``, ``pool``, and ``fc`` are defined in ``learn/modelGenerator.py``. And are here defined for the 3D case. More classes can be defined following the same format.


Deep learning
---------------

We are now all set to start the deep learning experiment. We are going to see how to set up both experiment for 2D and 3D case. By default the network performs a regression on the score requrested. However it is possible to specify a classification by just changing a few parameters


Regression with a 3D CNN
^^^^^^^^^^^^^^^^^^^^^^^^^^

The default options are all set to perform a regression using 3D volumetric data. Therefore we here simply need to create an instance of the ``NeuralNetwork`` class with options set to thir default values (i.e. we don't need to specify them):

>>> model = NeuralNet(data_set,cnn)

``data_set`` is the dataset created above and ``cnn`` is the automatically generated network. Other options can be specified here but that will do for now. Creating an instance of ``NeuralNet`` initialize all the required parts to do deep learning. The only thing we therefore need to do is to train the network

>>> model.train(nepoch = 50,divide_trainset=0.8, train_batch_size = 5,num_workers=0)

We specify here the number of epoch, the amount of data used for training (the remaining data is for validation 0.2 here), the batch size and the number of workers (CPU threads) in charge of batch preparation. This will start the training process and output regression plots and the corresponding data ``data.hdf5``.

Regression with a 2D CNN
^^^^^^^^^^^^^^^^^^^^^^^^^^

Deeprank also allows to transform the 3D volumetric data in 2D data by slicing planes of the data and using each plane as given channel. Very little modification of the code are necessary to do so. The creation of the dataset is identical to the 3D case, you must simply specify ``model_type=2D`` in the definition of the NeuralNet

>>> model = NeuralNet(data_set,cnn,model_type='2d',proj2d=0)

And that's it. The ``proj2d`` argumetn specify how to slice the 3D volumetric data. Value of: 0, 1, 2 are possible to slice along the YZ, XZ or XY plane respectively. Note that the ``cnn`` used here also must be a 2D CNN and not a 3D CNN.

Binary Classification
^^^^^^^^^^^^^^^^^^^^^^

If one want to perform a binary classification just a few modification must be performed. First the last fully connected layer of the network must have a size of 2. Hence make sure that the definition of the network is something like that

>>> class cnn(nn.Module):
>>>
>>>     def __init__(self,input_shape):
>>>         super(cnn,self).__init__()
>>> 
>>>         self.convlayer_000 = nn.Conv3d(input_shape[0],4,kernel_size=2)
>>>         ....
>>>         ....
>>>         self.fclayer_001 = nn.Linear(84,2)
>>> 
>>> 
>>>     def _get_conv_output(self,shape):
>>>         ...

Once this is done you simply have to set one option in the creation of the ``NeuralNetwork`` instance

>>> model = NeuralNet(data_set,cnn,task='reg')

And that's it really. Specifying ``task='reg'`` wil automatically adjust all the parameters of the training process to perform a regression. It will for example set the loss function to a cross entropy loss.

Reusing a pretrained model
---------------------------

In many cases after you've trained the network you would like to reuse the model either to test its performace on a test set or to continue the training. To do that you would also like to reuse the options for the dataset (i.e. the same feature, target, pairing of the features, etc ...). All of the can be done automatically with DeepRank and an example is given in ``test/notravis_test_transfer.py``. Let's say that the pretrained model (automatically generated at the end of the training) is located at ``model.pth.tar``. In that case you can simply specify the following:

>>> database = '1ak4.hdf5'
>>> model = NeuralNet(database,cnn,pretrained_model='model.pth.tar')
>>> model.test()

Note that here the database is simply the name of the hdf5 file we want to test the model on. All the processing of the dataset will be automatically done in the exact same way than it was done during the training of the model. Hence you do not have to copy the ``select_features`` and ``select_target`` .... arguments, all that is done for you.
