import os
import unittest
import glob
import numpy as np
try:
  from deeprank.learn import *
  from deeprank.learn.model3d import cnn as cnn3d
  from deeprank.learn.model3d import cnn_class as cnn3d_class
  from deeprank.learn.model2d import cnn as cnn2d
  skip=False
except:
  skip=True

# all the import torch fails on TRAVIS
# so we can only exectute this test locally
class TestLearn(unittest.TestCase):

  @unittest.skipIf(skip,"torch fails on Travis")
  @staticmethod
  def test_learn_3d_reg_mapfly():
    """Use a 3D CNN for regularization."""

    #adress of the database
    database = '1ak4.hdf5'
    if not os.path.isfile(database):
      raise FileNotFoundError('Database %s not found. Make sure to run test_generate before')

    # clean the output dir
    out = './out_3d'
    if os.path.isdir(out):
      for f in glob.glob(out+'/*'):
        os.remove(f)
      os.removedirs(out)

    # declare the dataset instance
    data_set = DataSet(database,
                test_database = None,
                mapfly = True,
                data_augmentation=None,
                grid_shape=(30,30,30),
                select_feature={'AtomicDensities' : {'CA':3.5, 'C':3.5, 'N':3.5, 'O':3.5},
                                'Features' : ['coulomb','vdwaals','charge','PSSM_*'] },
                select_target='DOCKQ',tqdm=True,
                normalize_features = True, normalize_targets=True,
                clip_features=False,
                pair_chain_feature=np.add,
                dict_filter={'IRMSD':'<4. or >10.'})


    # create the networkt
    model = NeuralNet(data_set,cnn3d,model_type='3d',task='reg',
                      cuda=False,plot=True,outdir=out)

    # start the training
    model.train(nepoch = 5,divide_trainset=0.8, train_batch_size = 5,num_workers=0)

  @unittest.skipIf(skip,"torch fails on Travis")
  @staticmethod
  def test_learn_3d_reg():
    """Use a 3D CNN for regularization."""

    #adress of the database
    database = '1ak4.hdf5'
    if not os.path.isfile(database):
      raise FileNotFoundError('Database %s not found. Make sure to run test_generate before')

    # clean the output dir
    out = './out_3d'
    if os.path.isdir(out):
      for f in glob.glob(out+'/*'):
        os.remove(f)
      os.removedirs(out)

    # declare the dataset instance
    data_set = DataSet(database,
                test_database = None,
                mapfly = False,
                data_augmentation=2,
                grid_shape=(30,30,30),
                select_feature={'AtomicDensities_ind' : 'all',
                                'Feature_ind' : ['coulomb','vdwaals','charge','PSSM_*'] },
                select_target='DOCKQ',tqdm=True,
                normalize_features = True, normalize_targets=True,
                clip_features=False,
                pair_chain_feature=np.add,
                dict_filter={'DOCKQ':'<1.'})
                #dict_filter={'IRMSD':'<4. or >10.'})


    # create the networkt
    model = NeuralNet(data_set,cnn3d,model_type='3d',task='reg',
                      cuda=False,plot=True,outdir=out)

    # start the training
    model.train(nepoch = 50,divide_trainset=0.8, train_batch_size = 5,num_workers=0, save_model='all')


  @unittest.skipIf(skip,"torch fails on Travis")
  @staticmethod
  def test_learn_2d_reg():
    """Use a 2D CNN for regularization."""

    #adress of the database
    database = '1ak4.hdf5'

    # clean the output dir
    out = './out_2d/'
    if os.path.isdir(out):
      for f in glob.glob(out+'/*'):
        os.remove(f)
      os.removedirs(out)

    if not os.path.isfile(database):
      raise FileNotFoundError('Database %s not found. Make sure to run test_generate before')

    # declare the dataset instance
    data_set = DataSet(database,
              test_database = database,
              select_feature={'AtomicDensities_ind' : 'all',
                              'Feature_ind' : ['coulomb','vdwaals','charge','pssm'] },
                              select_target='DOCKQ',tqdm=True,
                              normalize_features = True, normalize_targets=True,
                              clip_features=False,
                              pair_chain_feature=np.add,
                              dict_filter={'IRMSD':'<4. or >10.'})


    # create the network
    model = NeuralNet(data_set,cnn2d,model_type='2d',task='reg',
                      cuda=False,plot=True,outdir=out)

    # start the training
    model.train(nepoch = 50,divide_trainset=0.8, train_batch_size = 5,num_workers=0)

  @unittest.skipIf(skip,"torch fails on Travis")
  @staticmethod
  def test_transfer():

    #adress of the database
    database = '1ak4.hdf5'

    if not os.path.isfile(database):
      raise FileNotFoundError('Database %s not found. Make sure to run test_generate before')

    # clean the output dir
    out = './out_test/'
    if os.path.isdir(out):
      for f in glob.glob(out+'/*'):
        os.remove(f)
      os.removedirs(out)

    # create the network
    model_name = './out_3d/last_model.pth.tar'
    model = NeuralNet(database,cnn3d,pretrained_model=model_name,outdir=out)
    model.test()


  @unittest.skipIf(skip,"Torch fails on Travis")
  @staticmethod
  def test_learn_3d_class():
    """Use a 3D CNN for regularization."""

    #adress of the database
    database = ['1ak4.hdf5','native.hdf5']

    # clean the output dir
    out = './out_3d_class'
    if os.path.isdir(out):
      for f in glob.glob(out+'/*'):
        os.remove(f)
      os.removedirs(out)

    # declare the dataset instance
    data_set = DataSet(database,
                test_database = None,
                grid_shape=(30,30,30),
                select_feature={'AtomicDensities_ind' : 'all',
                                'Feature_ind' : ['coulomb','vdwaals','charge','PSSM_*'] },
                select_target='BIN_CLASS',tqdm=True,
                normalize_features = True, 
                normalize_targets=False,
                clip_features=False,
                pair_chain_feature=np.add)


    # create the networkt
    model = NeuralNet(data_set,cnn3d_class,model_type='3d',task='class',
                      cuda=False,plot=True,outdir=out)

    # start the training
    model.train(nepoch = 50,divide_trainset=0.8, train_batch_size = 5,num_workers=0,save_epoch='all')


if __name__ == "__main__":


  TestLearn.test_learn_3d_reg()
  #TestLearn.test_learn_3d_class()

  #TestLearn.test_learn_2d_reg()
  #TestLearn.test_transfer()