import pickle

import numpy as np

import deeprank.learn.modelGenerator
import torch.optim as optim
from deeprank.learn import NetworkGenerator, NeuralNet, DataSet


class saved_model(object):
    def __init__(
            self,
            conv_layers_params=None,
            fc_layers_params=None,
            reward=None):
        self.conv_layers_params = conv_layers_params
        self.fc_layers_params = fc_layers_params
        self.reward = reward


class MetaQNN(object):

    def __init__(self, final_dim=1):

        # names
        self.model_name = 'conv3d'
        self.file_name = 'model.py'

        # data storage
        self.model_generator = None
        self.memory = []

        # max number of layers
        self.num_conv_layers = range(1, 11)
        self.num_fc_layers = range(1, 5)

        # types of layers possible
        self.conv_types = ['conv', 'dropout', 'pool']

        # types of post processing
        # must be in torch.nn.functional
        self.post_types = [None, 'relu']

        # params of conv layers
        self.conv_params = {}
        self.conv_params['output_size'] = range(1, 10)
        self.conv_params['kernel_size'] = range(2, 5)

        # params of pool layers
        self.pool_params = {}
        self.pool_params['kernel_size'] = range(2, 5)

        # params of the dropout layers
        self.dropout_params = {}
        self.dropout_params['percent'] = np.linspace(0.1, 0.9, 9)

        # params of the fc layers
        self.fc_params = {}
        self.fc_params['output_size'] = [2**i for i in range(4, 11)]

        # store the current layers/reward
        self.conv_layers = []
        self.fc_layers = []
        self.reward = 0

        # dimension of the final layer
        self.final_dim = final_dim

        # guess the task (regression/classification)
        if self.final_dim == 1:
            self.task = 'reg'
        else:
            self.task = 'class'

    #########################################
    #
    # save the model to the class memory
    #
    #########################################
    def store_model(self):

        conv_layers_params = []
        for layer in self.conv_layers:
            conv_layers_params.append(layer.__get_params__())

        fc_layers_params = []
        for layer in self.fc_layers:
            fc_layers_params.append(layer.__get_params__())

        self.memory.append(saved_model(conv_layers_params=conv_layers_params,
                                       fc_layers_params=fc_layers_params),
                           reward=self.reward)

    #########################################
    #
    # save the the entire memory to disk
    #
    #########################################
    def pickle_memory(self, fname='memory.pkl'):
        pickle.dump(self.memory, open(fname, "wb"))

    #########################################
    #
    # write a model to file
    #
    #########################################
    def write_model(self):
        model_generator = NetworkGenerator(name=self.model_name,
                                           fname=self.file_name,
                                           conv_layers=self.conv_layers,
                                           fc_layers=self.fc_layers)
        model_generator.print()
        model_generator.write()

    #########################################
    #
    # get a new random model
    #
    #########################################
    def get_new_random_model(self):

        print('QNN: Generate new model')
        # number of conv/fc layers
        nconv = np.random.choice(self.num_conv_layers)
        nfc = np.random.choice(self.num_fc_layers)

        # generate the conv layers
        self.conv_layers = []
        for ilayer in range(nconv):
            self._init_conv_layer_random(ilayer)

        # generate the fc layers
        self.fc_layers = []
        for ilayer in range(nfc):
            self._init_fc_layer_random(ilayer)

        # fix the final dimension
        self.fc_layers[-1].output_size = self.final_dim

        # write the model to file
        self.write_model()

    # pick a layer type
    def _init_conv_layer_random(self, ilayer):

        # determine wih type of layer we want
        # first layer is a conv
        # we can't have 2 pool in a row
        if ilayer == 0:
            name = self.conv_types[0]

        # if rpevious layer is pool, next can't be pool
        elif self.conv_layers[ilayer - 1].__name__ == 'pool':
            name = np.random.choice(self.conv_types[:-1])

        # else it can be anything
        else:
            name = np.random.choice(self.conv_types)

        # init the parms of the layer
        # each layer type has its own params
        # the output/input size matching is done automatically
        if name == 'conv':
            params = {}
            params['name'] = name

            if ilayer == 0:
                params['input_size'] = -1  # fixed by input shape
            else:
                for isearch in range(ilayer - 1, -1, -1):
                    if self.conv_layers[isearch].__name__ == 'conv':
                        params['input_size'] = self.conv_layers[isearch].output_size
                        break

            params['output_size'] = np.random.choice(
                self.conv_params['output_size'])
            params['kernel_size'] = np.random.choice(
                self.conv_params['kernel_size'])
            params['post'] = np.random.choice(self.post_types)

        if name == 'pool':
            params = {}
            params['name'] = name
            params['kernel_size'] = np.random.choice(
                self.pool_params['kernel_size'])
            params['post'] = np.random.choice(self.post_types)

        if name == 'dropout':
            params = {}
            params['name'] = name
            params['percent'] = np.random.choice(
                self.dropout_params['percent'])

        # create the current layer class instance
        # and initialize if with the __init_from_dict__() method
        current_layer = getattr(
            deeprank.learn.modelGenerator,
            params['name'])()
        current_layer.__init_from_dict__(params)
        self.conv_layers.append(current_layer)

    def _init_fc_layer_random(self, ilayer):

        # init the parms of the layer
        # each layer type has its own params
        # the output/input size matching is done automatically
        name = 'fc'  # so far only fc layer here
        params = {}
        params['name'] = name
        if ilayer == 0:
            params['input_size'] = -1  # fixed by the conv layers
        else:
            params['input_size'] = self.fc_layers[ilayer - 1].output_size

        params['output_size'] = np.random.choice(self.fc_params['output_size'])
        params['post'] = np.random.choice(self.post_types)

        current_layer = getattr(
            deeprank.learn.modelGenerator,
            params['name'])()
        current_layer.__init_from_dict__(params)
        self.fc_layers.append(current_layer)

    # load the data set in memory only once
    def load_dataset(self, database, feature='all', target='DOCKQ'):

        print('QNN: Load data set')
        self.data_set = DataSet(database,
                                select_feature=feature,
                                select_target=target,
                                normalize_features=True,
                                normalize_targets=True)

        self.data_set.load()

    def train_model(self, cuda=False, ngpu=0):

        print('QNN: Train model')
        from .model3d import cnn

        # create the ConvNet
        model = NeuralNet(self.data_set, cnn, plot=False, cuda=cuda, ngpu=ngpu)

        # fix optimizer
        model.optimizer = optim.SGD(model.net.parameters(),
                                    lr=0.001, momentum=0.9, weight_decay=0.005)

        # train and save reward
        model.train(nepoch=20)
        self.reward = model.test_loss
