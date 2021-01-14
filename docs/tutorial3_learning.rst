Learning
========

This section describes how to prepare training/validation/test datasets with specific features and targets, how to configure neural network architecture, and how to train and test a neural network with DeepRank.

This tutorial uses the example from the test file ``test/test_learn.py``.

Let's start from importing the necessary DeepRank modules,

>>> from deeprank.learn import Dataset, NeuralNet

The ``Dataset`` module is used for preparing training/validation/test datasets, and the ``NeuralNet`` module for training and testing neural networks.

Creating training/validation/test datasets
------------------------------------------

First we need to provide the HDF5 file we generated in the `Data Generation`_ step,

.. _Data Generation: tutorial2_dataGeneration.html

>>> database = '1ak4.hdf5'

You can also provide multiple HDF5 files to a database, e.g.

>>> database = ['1ak4.hdf5','native.hdf5']

Then we will use the data from this ``database`` to create the training/validation/test  datasets,

>>> data_set = DataSet(train_database=database,
>>>                   valid_database=None,
>>>                   test_database=None,
>>>                   chain1='C',
>>>                   chain2='D',
>>>                   grid_shape=(30, 30, 30),
>>>                   select_feature={
>>>                     'AtomicDensities_ind': 'all',
>>>                     'Feature_ind': ['coulomb', 'vdwaals', 'charge','PSSM_*']},
>>>                    select_target='BIN_CLASS',
>>>                    normalize_features=True,
>>>                    normalize_targets=False,
>>>                    pair_chain_feature=np.add,
>>>                    dict_filter={'IRMSD':'<4. or >10.'})

Here we have only one database, so we must provide it to the ``train_database`` parameter, and later you will set how to split this database to training, validation and test data. When independent validation and test databases exist, you can then set the corresponding ``valid_database`` and ``test_database`` parameters.

You must also specify which features and targets to use for the neural network by setting the ``select_feature`` and ``select_target`` parameters. By default, all features exist in the HDF5 files will be selected.

When you specify some of the features, you have to use a *dictionary*. As you can see in the HDF5 file, the ``mapped_features`` subgroup of each complex contains two groups ``AtomicDensities_ind`` and ``Feature_ind``. We'll forget about the ``_ind`` that is for legacy reasons, but these two groups contains the atomic densities and other features respectively. Therefore the value used in the example above:

>>> select_feature={'AtomicDensities': 'all',
>>>                 'Features': ['coulomb', 'vdwaals', 'charge', 'PSSM_*']}

It means that we want to use all the atomic densities features, the ``coulomb``, ``vdwaals`` and ``charge`` features as well as the PSSM features with a name starting with ``PSSM_``.

With the ``normalize_features`` and ``normalize_targets`` parameters, you can use the normalized values of features or targets to train a neural network.

As you can see in the HDF5 file for each feature, the data of ``chain1`` and ``chain2`` are stored separately. But it is possible to combine the feature of both chains into a single channel via the parameter ``pair_chain_feature``, e.g.,

>>> pair_chain_feature = np.add

which means that feature value of ``chain1`` and ``chain2`` will be summed up to create a single channel.

You can further filter the data with specific conditions,

>>> dict_filter={'IRMSD':'<4. or >10.'}

it will only selects the complexes whose IRMSD are lower than 4Å or larger than 10Å.


Creating neural network architecture
------------------------------------

The architecture of the neural network has to be well defined before training. And DeepRank provides a very useful tool ``modelGenerator`` to facilitate this process.

Here is a simple example showing how to generate NN architecture,

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
>>> fc_layers.append(fc(input_size=84,output_size=2))
>>>
>>> gen = NetworkGenerator(name='cnn_demo',
>>>                       fname='cnn_arch_demo.py',
>>>                       conv_layers=conv_layers,
>>>                       fc_layers=fc_layers)
>>> gen.print()
>>> gen.write()

It will print out the human readable summary of the architecture,

.. code-block::

    #----------------------------------------------------------------------
    # Network Structure
    #----------------------------------------------------------------------
    #conv layer   0: conv | input -1  output  4  kernel  2  post relu
    #conv layer   1: pool | kernel  2  post None
    #conv layer   2: conv | input  4  output  5  kernel  2  post relu
    #conv layer   3: pool | kernel  2  post None
    #fc   layer   0: fc   | input -1  output  84  post relu
    #fc   layer   1: fc   | input  84  output  2  post None
    #----------------------------------------------------------------------

and also generate a ``cnn_arch_demo.py`` file with a class ``cnn_demo`` that defines the NN architecture.

We also provide the predefined NN architectures in ``learn/model3d.py`` and ``learn/model2d.py``.

Training a neural network
-------------------------

We are now all set to start the deep learning experiments. DeepRank supports both classification and regression tasks.


Classification with 3D CNN
^^^^^^^^^^^^^^^^^^^^^^^^^^

>>> from deeprank.learn.model3d import cnn_class
>>>
>>> model = NeuralNet(data_set=data_set,
>>>                  model=cnn_class,
>>>                  model_type='3d',
>>>                  task='class')

``data_set`` is the dataset created above and ``cnn_class`` is the predefined NN architecture.
We also need to specify the ``model_type`` and the learning ``task``.

Then we can start the training process,

>>> model.train(nepoch=50,
>>>             divide_trainset=[0.7, 0.2, 0.1]
>>>             train_batch_size=5,
>>>             num_workers=1,
>>>             hdf5='epoch_data_class.hdf5')

We specify here the number of epoch, the fraction of data for training/validation/test sets, the batch size and the number of workers (CPU threads) in charge of batch preparation, and the output HDF5 file for training results. The model will be save to ``.pth.tar`` files, e.g. ``model_epoch_0001.pth.tar``.

Regression with 3D CNN
^^^^^^^^^^^^^^^^^^^^^^

To train a regression model, the steps are same as the classification above. But you need to provide the regression NN architecture and set the correct task type, e.g.

>>> from deeprank.learn.model3d import cnn_reg
>>>
>>> model = NeuralNet(data_set=data_set,
>>>                  model=cnn_reg,
>>>                  model_type='3d',
>>>                  task='reg')
>>>
>>> model.train(nepoch=50,
>>>             divide_trainset=[0.7, 0.2, 0.1]
>>>             train_batch_size=5,
>>>             num_workers=1,
>>>             hdf5='epoch_data_reg.hdf5')

2D CNN
^^^^^^

DeepRank allows to transform the 3D volumetric data to 2D data by slicing planes of the data and using each plane as given channel. Very little modification of the code are necessary to do so. The creation of the dataset is identical to the 3D case, and you must specify ``model_type=2d`` for ``NeuralNet``,

>>> from deeprank.learn.model2d import cnn
>>>
>>> model = NeuralNet(data_set=data_set_2d,
>>>                  model=cnn,
>>>                  model_type='2d',
>>>                  task='reg',
>>>                  proj2d=0)

The ``proj2d`` parameter defines how to slice the 3D volumetric data. Value of: 0, 1, 2 are possible to slice along the YZ, XZ or XY plane respectively.

Testing a neural network
------------------------

In many cases after you've trained the NN model, you would like to use the model to do prediction or to test the model's performance on new data. DeepRank provide a very easy way to do that, let's say we have got the trained classification model ``model.pth.tar``,

>>> from deeprank.learn.model3d import cnn_class
>>>
>>> database = '1AK4_test.hdf5'
>>> model = NeuralNet(database, cnn_class, pretrained_model='model.pth.tar')
>>> model.test()

Note that here the database is simply the name of the hdf5 file we want to test the model on. All the processing of the dataset will be automatically done in the exact same way as it was done during the training of the model. Hence you do not have to copy the ``select_features`` and ``select_target`` parameters, all that is done for you automatically.
