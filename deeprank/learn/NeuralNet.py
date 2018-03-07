#!/usr/bin/env python
import sys
import os
import time
import h5py
import matplotlib.pyplot as plt
import numpy as np

#import torch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

# cuda
import torch.cuda

# dataset
from deeprank.learn import DataSet



class NeuralNet():

    def __init__(self,data_set,model,
                 model_type='3d',proj2d=0,task='reg',
                 pretrained_model=None,
                 cuda=False,ngpu=0,
                 plot=True,outdir='./'):

        """Train a Convolutional Neural Network for DeepRank.

        Example:

        >>> # create the network
        >>> model = NeuralNet(data_set,cnn,model_type='3d',task='reg',
        >>>                   cuda=False,plot=True,outdir='./out/')
        >>>
        >>> # start the training
        >>> model.train(nepoch = 50,divide_trainset=0.8, train_batch_size = 5,num_workers=0)

        Attributes:

            data_set (deeprank.dataset or str):  Data set used for training or testing

                data_set = DeepRankDataSet( ... ) for training

                data_set = 'xxx.hdf5' when pretrained model is loaded

            model (nn.Module): Definition of the NN to use. Must subclass nn.Module.
                See examples in model2D.py and model3d.py

            model_type (srt): Type of model we want to use. Must be '2d' or '3d'.
                If we specify a 2d model, the data set is automatically converted
                to the correct format.

            task (str 'ref' or 'class'): Task to perform:
                reg' for regression, 'class' for classification
                The loss function, the datatype of the targets and plot functions
                will be autmatically adjusted depending on the task

            plot (bool): Plot the results

            outdir (str): output directory where all the files will be written

            pretrained_model (str): Save model to be used for further training or testing

            cuda (bool): Use CUDA

            ngpu (int): number of GPU to be used

            Raises:
                ValueError: if dataset format is not recognized
            """

        #------------------------------------------
        # Dataset
        #------------------------------------------

        #data set and model
        self.data_set = data_set

        # pretrained model
        self.pretrained_model = pretrained_model

        if isinstance(data_set,(str,list)) and pretrained_model is None:
            raise ValueError('Argument data_set must be a DeepRankDataSet object\
                              when no pretrained model is loaded')

        # load the model
        if self.pretrained_model is not None:

            # create the dataset if required
            # but don't process it yet
            if isinstance(self.data_set,str) or isinstance(self.data_set,list):
                self.data_set = DataSet(self.data_set,process=False)

            # load the model and
            # change dataset parameters
            self.load_data_params(self.pretrained_model)

            # process it
            self.data_set.process_dataset()

        # convert the data to 2d if necessary
        if model_type == '2d':

            self.data_set.transform = True
            self.data_set.proj2D = proj2d
            self.data_set.get_input_shape()


        #------------------------------------------
        # CUDA
        #------------------------------------------

        # CUDA required
        self.cuda = cuda
        self.ngpu = ngpu

        # handles GPU/CUDA
        if self.ngpu > 0:
            self.cuda = True

        if self.ngpu == 0 and self.cuda :
            self.ngpu = 1


        #------------------------------------------
        # Regression or classifiation
        #------------------------------------------

        # task to accomplish
        self.task = task

        # Set the loss functiom
        if self.task=='reg':
            self.criterion = nn.MSELoss(size_average=False)
            self._plot_scatter = self._plot_scatter_reg

        elif self.task=='class':
            self.criterion = nn.CrossEntropyLoss()
            self._plot_scatter = self._plot_boxplot_class

        else:
            raise ValueError("Task " + self.task +"not recognized.\nOptions are \n\t 'reg': regression \n\t 'class': classifiation\n\n")

        #------------------------------------------
        # Output
        #------------------------------------------

        # plot or not plot
        self.plot = plot

        # output directory
        self.outdir = outdir
        if self.plot:
            if not os.path.isdir(self.outdir):
                os.mkdir(outdir)

        #------------------------------------------
        # Network
        #------------------------------------------

        # load the model
        self.net = model(self.data_set.input_shape)

        #multi-gpu
        if self.ngpu>1:
            ids = [i for i in range(self.ngpu)]
            self.net = nn.DataParallel(self.net,device_ids=ids).cuda()

        # cuda compatible
        elif self.cuda:
            self.net = self.net.cuda()

        # set the optimizer
        self.optimizer = optim.SGD(self.net.parameters(),lr=0.005,momentum=0.9,weight_decay=0.001)

        # laod the parameters of the model if provided
        if self.pretrained_model:
            self.load_model_params(self.pretrained_model)

        #------------------------------------------
        # print
        #------------------------------------------

        print('\n')
        print('='*40)
        print('=\t Convolution Neural Network')
        print('=\t model     : %s' %model_type)
        print('=\t CNN       : %s' %model.__name__)

        for feat_type,feat_names in self.data_set.select_feature.items():
            print('=\t features  : %s' %(feat_type))
            for name in feat_names:
                print('=\t\t     %s' %(name))
        if self.data_set.pair_chain_feature is not None:
            print('=\t Pair      : %s' %self.data_set.pair_chain_feature.__name__)
        print('=\t targets   : %s' %self.data_set.select_target)
        print('=\t CUDA      : %s' %str(self.cuda))
        if self.cuda:
            print('=\t nGPU      : %d' %self.ngpu)
        print('='*40,'\n')

        # check if CUDA works
        if self.cuda and not torch.cuda.is_available():
            print(' --> CUDA not deteceted : Make sure that CUDA is installed and that you are running on GPUs')
            print(' --> To turn CUDA of set cuda=False in NeuralNet')
            print(' --> Aborting the experiment \n\n')
            sys.exit()

    def train(self,nepoch=50, divide_trainset=None, hdf5='data.hdf5',train_batch_size = 10,
              preshuffle = True,export_intermediate=True,num_workers=1,save_model_name = 'model.pth.tar'):

        """Perform a simple training of the model. The data set is divided in training/validation sets.

        Args:

            nepoch (int, optional): number of iterations to go through the training

            divide_trainset (None, optional): the percentage assign to the training, validation and test set

            hdf5 (str, optional): file to store the training results

            train_batch_size (int, optional): size of the batch

            preshuffle (bool, optional): preshuffle the dataset before dividing it

            export_intermediate (bool, optional): export data at interediate epoch

            num_workers (int, optional): number of workers to be used to prep the batch data

        Example :

        >>> # declare the dataset instance
        >>> data_set = DataSet(database,
        >>>                           test_database = None,
        >>>                           grid_shape=(30,30,30),
        >>>                           select_feature={'AtomicDensities_ind' : 'all',
        >>>                                           'Feature_ind' : ['coulomb','vdwaals','charge','pssm'] },
        >>>                           select_target='IRMSD',tqdm=True,
        >>>                           normalize_features = True, normalize_targets=True,clip_features=True)
        >>>                           #pair_chain_feature=np.add,
        >>>                           #dict_filter={'IRMSD':'<4. or >10.'})
        >>> # create the network
        >>> model = NeuralNet(data_set,cnn,model_type='3d',task='reg',
        >>>                   cuda=False,plot=True,outdir='./out/')
        >>> # start the training
        >>> model.train(nepoch = 50,divide_trainset=0.8, train_batch_size = 5,num_workers=0)
        >>> # save the model
        >>> model.save_model()

        """

        # multi-gpu
        if self.ngpu > 1:
            train_batch_size *= self.ngpu

        print('\n: Batch Size : %d' %train_batch_size)
        if self.cuda:
            print(': NGPU       : %d' %self.ngpu)

        # hdf5 support
        fname =self.outdir+'/'+hdf5
        self.f5 = h5py.File(fname,'w')

        # divide the set in train+ valid and test
        divide_trainset = divide_trainset or [0.8,0.2]
        index_train,index_valid,index_test = self._divide_dataset(divide_trainset,preshuffle)

        print(': %d confs. for training' %len(index_train))
        print(': %d confs. for validation' %len(index_valid))
        print(': %d confs. for testing' %len(index_test))

        # train the model
        t0 = time.time()
        self._train(index_train,index_valid,index_test,
                    nepoch=nepoch,
                    train_batch_size=train_batch_size,
                    export_intermediate=export_intermediate,
                    num_workers=num_workers)
        self.f5.close()
        print(' --> Training done in ', time.strftime('%H:%M:%S', time.gmtime(time.time()-t0)))

        # save the model
        self.save_model(filename=save_model_name)

    def test(self,hdf5='test_data.hdf5'):
        """Test a predefined model on a new dataset.

        Example:
            >>> # adress of the database
            >>> database = '1ak4.hdf5'
            >>> # Load the model in a new network instance
            >>> model = NeuralNet(database,cnn,pretrained_model='./out/model.pth.tar',outdir='./test/')
            >>> # test the model
            >>> model.test()

        Args:
            hdf5 (str, optional): name of the hdf5 file to store the data

        """

        # hdf5 support
        fname =self.outdir+'/'+hdf5
        self.f5 = h5py.File(fname,'w')

        index = list(range(self.data_set.__len__()))
        sampler = data_utils.sampler.SubsetRandomSampler(index)
        loader = data_utils.DataLoader(self.data_set,sampler=sampler)
        self.data = {}
        _,self.data['test'] = self._epoch(loader,train_model=False)
        self._plot_scatter_reg(self.outdir+'/prediction.png')
        self.plot_hit_rate(self.outdir+'/hitrate.png')
        self._export_epoch_hdf5(0,self.data)
        self.f5.close()

    def save_model(self,filename='model.pth.tar'):

        """save the model to disk

        Args:
            filename (str, optional): name of the file
        """
        filename = self.outdir + '/' + filename

        state = {'state_dict'   : self.net.state_dict(),
                'optimizer'    : self.optimizer.state_dict(),
                'normalize_targets' : self.data_set.normalize_targets,
                'normalize_features': self.data_set.normalize_features,
                'select_feature' : self.data_set.select_feature,
                'select_target' : self.data_set.select_target,
                'pair_chain_feature' : self.data_set.pair_chain_feature,
                'dict_filter' : self.data_set.dict_filter,
                'transform'   : self.data_set.transform,
                'proj2D'      : self.data_set.proj2D}

        if self.data_set.normalize_features:
            state['feature_mean'] =  self.data_set.feature_mean
            state['feature_std' ] = self.data_set.feature_std

        if self.data_set.normalize_targets:
            state['target_min']  = self.data_set.target_min
            state['target_max']  = self.data_set.target_max

        torch.save(state,filename)

    def load_model_params(self,filename):
        """Load a saved model.

        Args:
            filename (str): filename
        """

        state = torch.load(filename)
        self.net.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])


    def load_data_params(self,filename):

        '''Load the parameters of the dataset.

        Args:
            filename (str): filename
        '''
        state = torch.load(filename)

        self.data_set.select_feature = state['select_feature']
        self.data_set.select_target  = state['select_target']

        self.data_set.pair_chain_feature = state['pair_chain_feature']
        self.data_set.dict_filter = state['dict_filter']

        self.data_set.normalize_targets = state['normalize_targets']
        if self.data_set.normalize_targets:
            self.data_set.target_min = state['target_min']
            self.data_set.target_max = state['target_max']

        self.data_set.normalize_features = state['normalize_features']
        if self.data_set.normalize_features:
            self.data_set.feature_mean = state['feature_mean']
            self.data_set.feature_std = state['feature_std']

        self.data_set.transform = state['transform']
        self.data_set.proj2D = state['proj2D']

    def _divide_dataset(self,divide_set, preshuffle):

        '''Divide the data set in a training validation and test according to the percentage in divide_set.

        Args:
            divide_set (list(float)): percentage used for training/validation/test
            preshuffle (bool): shuffle the dataset before dividing it

        Returns:
            list(int),list(int),list(int): Indices of the training/validation/test set
        '''
        # if user only provided one number
        # we assume it's the training percentage
        if not isinstance(divide_set,list):
            divide_set = [divide_set, 1.-divide_set]

        # if user provided 3 number and testset
        if len(divide_set) == 3 and self.data_set.test_database is not None:
            divide_set = [divide_set[0],1.-divide_set[0]]
            print('   : test data set AND test in training set detected')
            print('   : Divide training set as %f train %f valid' %(divide_set[0],divide_set[1]))
            print('   : Keep test set for testing')


        # preshuffle
        if preshuffle:
            np.random.shuffle(self.data_set.index_train)

        # size of the subset for training
        ntrain = int( np.ceil(float(self.data_set.ntrain)*divide_set[0]) )
        nvalid = int( np.floor(float(self.data_set.ntrain)*divide_set[1]) )

        # indexes train and valid
        index_train = self.data_set.index_train[:ntrain]
        index_valid = self.data_set.index_train[ntrain:ntrain+nvalid]

        # index of test depending of the situation
        if len(divide_set)==3:
            index_test = self.data_set.index_train[ntrain+nvalid:]
        else:
            index_test = self.data_set.index_test

        return index_train,index_valid,index_test



    def _train(self,index_train,index_valid,index_test,
               nepoch = 50,train_batch_size = 5,
               export_intermediate=False,num_workers=1):

        """Train the model.

        Args:
            index_train (list(int)): Indices of the training set
            index_valid (list(int)): Indices of the validation set
            index_test  (list(int)): Indices of the testing set
            nepoch (int, optional): numbr of epoch
            train_batch_size (int, optional): size of the batch
            export_intermediate (bool, optional):export itnermediate data
            num_workers (int, optional): number of workers pytorch uses to create the batch size

        Returns:
            torch.tensor: Parameters of the network after training
        """

        # printing options
        nprint = np.max([1,int(nepoch/10)])

        # store the length of the training set
        ntrain = len(index_train)

        # pin memory for cuda
        pin = False
        if self.cuda:
            pin = True

        # create the sampler
        train_sampler = data_utils.sampler.SubsetRandomSampler(index_train)
        valid_sampler = data_utils.sampler.SubsetRandomSampler(index_valid)
        test_sampler = data_utils.sampler.SubsetRandomSampler(index_test)

        # get if we test as well
        _test_ = len(test_sampler.indices)>0

        # containers for the losses
        self.losses={'train': [],'valid': []}
        if _test_:
            self.losses['test'] = []


        #  create the loaders
        train_loader = data_utils.DataLoader(self.data_set,batch_size=train_batch_size,sampler=train_sampler,pin_memory=pin,num_workers=num_workers,shuffle=False,drop_last=False)
        valid_loader = data_utils.DataLoader(self.data_set,batch_size=train_batch_size,sampler=valid_sampler,pin_memory=pin,num_workers=num_workers,shuffle=False,drop_last=False)

        if _test_:
            test_loader = data_utils.DataLoader(self.data_set,batch_size=train_batch_size,sampler=test_sampler,pin_memory=pin,num_workers=num_workers,shuffle=False,drop_last=False)

        # training loop
        av_time = 0.0
        self.data = {}
        for epoch in range(nepoch):

            print('\n: epoch %03d / %03d ' %(epoch,nepoch) + '-'*45)
            t0 = time.time()

            # we valid/test before training so that the data for the three
            # are calcualted with the same NN

            # validate the model
            self.valid_loss,self.data['valid'] = self._epoch(valid_loader,train_model=False)
            self.losses['valid'].append(self.valid_loss)

            # test the model
            if _test_:
                test_loss,self.data['test'] = self._epoch(test_loader,train_model=False)
                self.losses['test'].append(test_loss)

            # process train data without training
            #self.train_loss,self.data['train'] = self._epoch(train_loader,train_model=False    )
            #self.losses['train'].append(self.train_loss)

            # train the model
            self.train_loss,self.data['train'] = self._epoch(train_loader,train_model=True)
            self.losses['train'].append(self.train_loss)

            # talk a bit about losse
            print('  train loss       : %1.3e\n  valid loss       : %1.3e' %(self.train_loss, self.valid_loss))
            if _test_:
                print('  test loss        : %1.3e' %(test_loss))

            # timer
            elapsed = time.time()-t0
            if elapsed>10:
                print('  epoch done in    :', time.strftime('%H:%M:%S', time.gmtime(elapsed)))
            else:
                print('  epoch done in    : %1.3f' %elapsed)

            # remaining time
            av_time += elapsed
            nremain = nepoch-(epoch+1)
            remaining_time = av_time/(epoch+1)*nremain
            print('  remaining time   :',  time.strftime('%H:%M:%S', time.gmtime(remaining_time)))

            # plot the scatter plots
            if (export_intermediate and epoch%nprint == nprint-1) or epoch==0 or epoch==nepoch-1:
                if self.plot:
                    figname = self.outdir+"/prediction_%03d.png" %epoch
                    self._plot_scatter(figname)

                    figname = self.outdir+"/hitrate_%03d.png" %epoch
                    self.plot_hit_rate(figname)

                self._export_epoch_hdf5(epoch,self.data)

            sys.stdout.flush()

        # plot the losses
        self._export_losses(self.outdir+'/'+'losses.png')

        return torch.cat([param.data.view(-1) for param in self.net.parameters()],0)

    def _epoch(self,data_loader,train_model):

        """Perform one single epoch iteration over a data loader.

        Args:
            data_loader (torch.DataLoader): DataLoader for the epoch
            train_model (bool): train the model if True or not if False

        Returns:
            float: loss of the model
            dict:  data of the epoch
        """

        running_loss = 0
        data = {'outputs':[],'targets':[],'mol':[]}
        n = 0
        debug_time = False
        time_learn = 0

        for d in data_loader:

            # get the data
            inputs = d['feature']
            targets = d['target']
            mol = d['mol']

            # transform the data
            inputs,targets = self._get_variables(inputs,targets)

            # zero gradient
            tlearn0 = time.time()

            # forward + loss
            outputs = self.net(inputs)
            loss = self.criterion(outputs,targets)
            running_loss += loss.data[0]
            n += len(inputs)

            # zero + backward + step
            if train_model:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            time_learn += time.time()-tlearn0

            # get the outputs for export
            if self.cuda:
                data['outputs'] +=  outputs.data.cpu().numpy().tolist()
                data['targets'] += targets.data.cpu().numpy().tolist()
            else:
                data['outputs'] +=  outputs.data.numpy().tolist()
                data['targets'] += targets.data.numpy().tolist()

            fname,molname = mol[0],mol[1]
            data['mol'] += [ (f,m) for f,m in zip(fname,molname)]

        # transform the output back
        if self.data_set.normalize_targets:
            data['outputs']  = self.data_set.backtransform_target(np.array(data['outputs']).flatten())
            data['targets']  = self.data_set.backtransform_target(np.array(data['targets']).flatten())
        else:
            data['outputs']  = np.array(data['outputs']).flatten()
            data['targets']  = np.array(data['targets']).flatten()

        # make np for export
        data['mol'] = np.array(data['mol'],dtype=object)

        # normalize the loss
        running_loss /= n

        return running_loss, data


    def _get_variables(self,inputs,targets):

        ''' Convert the feature/target in torch.Variables.

        The format is different for regression where the targets are float
        and classification where they are int.

        Args:
            inputs (np.array): raw features
            targets (np.array): raw target values

        Returns:
            torch.Variable: features
            torch.Variable: target values
        '''

        # if cuda is available
        if self.cuda:
            inputs = inputs.cuda(async=True)
            targets = targets.cuda(async=True)


        # get the varialbe as float by default
        inputs,targets = Variable(inputs).float(),Variable(targets).float()

        # change the targets to long for classification
        if self.task == 'class':
            targets =  targets.long()

        return inputs,targets


    def _export_losses(self,figname):

        '''Plot the losses vs the epoch

        Args:
            figname (str): name of the file where to export the figure
        '''

        print('\n --> Loss Plot')

        color_plot = ['red','blue','green']
        labels = ['Train','Valid','Test']

        fig,ax = plt.subplots()
        for ik,name in enumerate(self.losses):
            plt.plot(np.array(self.losses[name]),c=color_plot[ik],label=labels[ik])

        legend = ax.legend(loc='upper left')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Losses')

        fig.savefig(figname)
        plt.close()

        grp = self.f5.create_group('/losses/')
        grp.attrs['type'] = 'losses'
        for k,v in self.losses.items():
            grp.create_dataset(k,data=v)

    def _plot_scatter_reg(self,figname):

        '''Plot a scatter plots of predictions VS targets.

        Useful to visualize the performance of the training algorithm

        Args:
            figname (str): filename

        '''

        # abort if we don't want to plot
        if self.plot is False:
            return


        print('\n --> Scatter Plot : ', figname, '\n')

        color_plot = {'train':'red','valid':'blue','test':'green'}
        labels = ['train','valid','test']

        fig,ax = plt.subplots()

        xvalues = np.array([])
        yvalues = np.array([])

        for l in labels:

            if l in self.data:

                targ = self.data[l]['targets']
                out = self.data[l]['outputs']

                xvalues = np.append(xvalues,targ)
                yvalues = np.append(yvalues,out)

                ax.scatter(targ,out,c = color_plot[l],label=l)

        legend = ax.legend(loc='upper left')
        ax.set_xlabel('Targets')
        ax.set_ylabel('Predictions')

        values = np.append(xvalues,yvalues)
        border = 0.1 * (values.max()-values.min())
        ax.plot([values.min()-border,values.max()+border],[values.min()-border,values.max()+border])

        fig.savefig(figname)
        plt.close()


    def plot_hit_rate(self,figname,irmsd_thr = 4.0):

        '''Plot the hit rate of the different training/valid/test sets

        The hit rate is defined as:
            the percentage of positive decoys that are included among the top m decoys.
            a positive decoy is a native-like one with a i-rmsd < 4A

        Args:
            figname (str): filename for the plot
            irmsd_thr (float, optional): threshold for 'good' models

        '''

        if self.plot is False:
            return

        print('\n --> Hit Rate :', figname, '\n')

        color_plot = {'train':'red','valid':'blue','test':'green'}
        labels = ['train','valid','test']

        fig,ax = plt.subplots()

        # get the target ordering
        inverse = self.data_set.target_ordering == 'lower'

        for l in labels:

            if l in self.data:

                # get the target values
                out = self.data[l]['outputs']

                # get the irmsd
                irmsd = []
                for fname,mol in self.data[l]['mol']:

                    f5 = h5py.File(fname,'r')
                    irmsd.append(f5[mol+'/targets/IRMSD'].value)
                    f5.close()

                # sort the data
                ind_sort = np.argsort(out)
                if not inverse:
                    ind_sort = ind_sort[::-1]
                irmsd = np.array(irmsd)[ind_sort]

                # compute the hit rate
                npos = len(irmsd[irmsd<irmsd_thr])
                if npos == 0:
                    npos = len(irmsd)
                    print('Warning : Non positive decoys found in %s for hitrate plot' % l)
                hit = np.cumsum(irmsd<irmsd_thr)/ npos

                # plot
                plt.plot(hit,c = color_plot[l],label=l)

        legend = ax.legend(loc='upper left')
        ax.set_xlabel('Top M')
        ax.set_ylabel('Hit Rate')

        fig.savefig(figname)
        plt.close()



    def _export_epoch_hdf5(self,epoch,data):
        """Export the epoch data to the hdf5 file.

        Export the data of a given epoch in train/valid/test group.
        In each group are stored the predcited values (outputs), ground truth (targets) and molecule name (mol)

        Args:
            epoch (int): index of the epoch
            data (dict): data of the epoch
        """

        # create a group
        grp_name = 'epoch_%04d' %epoch
        grp = self.f5.create_group(grp_name)

        # create attribute for DeepXplroer
        grp.attrs['type'] = 'epoch'

        # loop over the pass_type : train/valid/test
        for pass_type,pass_data in data.items():

            # we don't want to breack the process in case of issue
            try:

                # create subgroup for the pass
                sg = grp.create_group(pass_type)

                # loop over the data : target/output/molname
                for data_name,data_value in pass_data.items():

                    # mol name is a bit different
                    # since there are strings
                    if data_name == 'mol':
                        string_dt = h5py.special_dtype(vlen=str)
                        sg.create_dataset(data_name,data=data_value,dtype=string_dt)

                    # output/target values
                    else:
                        sg.create_dataset(data_name,data=data_value)

            except TypeError:
                print('Epoch Error export')
