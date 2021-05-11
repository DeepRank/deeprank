import glob
import os
import unittest

import numpy as np

try:
    from deeprank.learn import *
    from deeprank.learn.model3d import cnn_reg as cnn3d
    from deeprank.learn.model3d import cnn_class as cnn3d_class
    from deeprank.learn.model2d import cnn as cnn2d
    skip = False
except BaseException:
    skip = True


# all the import torch fails on TRAVIS
# so we can only exectute this test locally
class TestLearn(unittest.TestCase):

    @unittest.skipIf(skip, "torch fails on Travis")
    @staticmethod
    def test_learn_3d_reg_mapfly():
        """Use a 3D CNN for regularization."""

        # adress of the database
        database = '1ak4_mapfly.hdf5'
        if not os.path.isfile(database):
            raise FileNotFoundError(
                'Database %s not found. Make sure to run test_generate before')

        # clean the output dir
        out = './out_3d_fly'
        if os.path.isdir(out):
            for f in glob.glob(out + '/*'):
                os.remove(f)
            os.removedirs(out)

        # declare the dataset instance
        data_set = DataSet(
            database,
            test_database=None,
            chain1='C',
            chain2='D',
            mapfly=True,
            use_rotation=2,
            grid_info={
                'number_of_points': (10, 10, 10),
                'resolution': (3, 3, 3)},
            select_feature={
                'AtomicDensities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8},
                'Features': ['coulomb', 'vdwaals', 'charge', 'PSSM_*']},
            select_target='DOCKQ',
            tqdm=True,
            normalize_features=False,
            normalize_targets=False,
            clip_features=False,
            pair_chain_feature=np.add,
            dict_filter={'DOCKQ': '<1'})
        # dict_filter={'IRMSD':'<4. or >10.'})

        # create the networkt
        model = NeuralNet(data_set, cnn3d, model_type='3d', task='reg',
                          cuda=False, plot=True, outdir=out)

        # start the training
        model.train(
            nepoch=5,
            divide_trainset=0.8,
            train_batch_size=2,
            preshuffle_seed=2019,
            num_workers=0)

    @unittest.skipIf(skip, "torch fails on Travis")
    @staticmethod
    def test_learn_3d_reg():
        """Use a 3D CNN for regularization."""

        # adress of the database
        train_database = '1ak4.hdf5'
        if not os.path.isfile(train_database):
            raise FileNotFoundError(
                'Database %s not found. Make sure to run test_generate before',
                train_database)

        # clean the output dir
        out = './out_3d_reg'
        if os.path.isdir(out):
            for f in glob.glob(out + '/*'):
                os.remove(f)
            os.removedirs(out)

        # declare the dataset instance
        data_set = DataSet(
            train_database=train_database,
            valid_database=None,
            test_database=None,
            chain1='C',
            chain2='D',
            mapfly=False,
            use_rotation=2,
            grid_shape=(30, 30, 30),
            select_feature={
                'AtomicDensities_ind': 'all',
                'Feature_ind': [ 'coulomb', 'vdwaals', 'charge', 'PSSM_*']},
            select_target='DOCKQ',
            tqdm=True,
            normalize_features=True,
            normalize_targets=True,
            clip_features=False,
            pair_chain_feature=np.add,
            dict_filter={ 'DOCKQ': '<1.'})
        # dict_filter={'IRMSD':'<4. or >10.'})

        # create the networkt
        model = NeuralNet(data_set, cnn3d, model_type='3d', task='reg',
                          cuda=False, plot=True, outdir=out)

        # start the training
        model.train(
            nepoch=5,
            divide_trainset=0.8,
            train_batch_size=2,
            num_workers=0,
            preshuffle_seed=2019,
            save_model='all')

    @unittest.skipIf(skip, "Torch fails on Travis")
    @staticmethod
    def test_learn_3d_class():
        """Use a 3D CNN for regularization."""

        # adress of the database
        database = ['1ak4.hdf5', 'native.hdf5']

        # clean the output dir
        out = './out_3d_class'
        if os.path.isdir(out):
            for f in glob.glob(out + '/*'):
                os.remove(f)
            os.removedirs(out)

        # declare the dataset instance
        data_set = DataSet(
            train_database=database,
            valid_database=None,
            test_database=None,
            chain1='C',
            chain2='D',
            mapfly=False,
            grid_shape=( 30, 30, 30),
            select_feature={
                'AtomicDensities_ind': 'all',
                'Feature_ind': [ 'coulomb', 'vdwaals', 'charge', 'PSSM_*']},
            select_target='BIN_CLASS',
            tqdm=True,
            normalize_features=True,
            normalize_targets=False,
            clip_features=False,
            pair_chain_feature=np.add)

        # create the networkt
        model = NeuralNet(data_set, cnn3d_class, model_type='3d', task='class',
                          cuda=False, plot=True, outdir=out)

        # start the training
        model.train(
            nepoch=5,
            divide_trainset=0.8,
            train_batch_size=2,
            num_workers=0,
            save_epoch='all')

    @unittest.skipIf(skip, "torch fails on Travis")
    @staticmethod
    def test_learn_2d_reg():
        """Use a 2D CNN for regularization."""

        # adress of the database
        database = '1ak4.hdf5'

        # clean the output dir
        out = './out_2d/'
        if os.path.isdir(out):
            for f in glob.glob(out + '/*'):
                os.remove(f)
            os.removedirs(out)

        if not os.path.isfile(database):
            raise FileNotFoundError(
                'Database %s not found. Make sure to run test_generate before')

        # declare the dataset instance
        data_set = DataSet(
            train_database=database,
            valid_database=None,
            test_database=None,
            chain1='C',
            chain2='D',
            mapfly=False,
            select_feature={
                'AtomicDensities_ind': 'all',
                'Feature_ind': [ 'coulomb', 'vdwaals', 'charge', 'PSSM_*']},
            select_target='DOCKQ',
            tqdm=True,
            normalize_features=True,
            normalize_targets=True,
            clip_features=False,
            pair_chain_feature=np.add,
            dict_filter={
                'IRMSD': '<4. or >10.'})

        # create the network
        model = NeuralNet(data_set, cnn2d, model_type='2d', task='reg',
                          cuda=False, plot=True, outdir=out)

        # start the training
        model.train(
            nepoch=5,
            divide_trainset=0.8,
            train_batch_size=2,
            num_workers=0)

    @unittest.skipIf(skip, "torch fails on Travis")
    @staticmethod
    def test_transfer():

        # adress of the database
        database = '1ak4.hdf5'

        if not os.path.isfile(database):
            raise FileNotFoundError(
                'Database %s not found. Make sure to run test_generate before')

        # clean the output dir
        out = './out_test/'
        if os.path.isdir(out):
            for f in glob.glob(out + '/*'):
                os.remove(f)
            os.removedirs(out)

        # create the network
        model_name = './out_3d_fly/last_model.pth.tar'
        model = NeuralNet(
            database,
            cnn3d,
            pretrained_model=model_name,
            chain1='C',
            chain2='D',
            outdir=out)
        model.test()


if __name__ == "__main__":

    TestLearn.test_learn_3d_reg_mapfly()
    TestLearn.test_learn_3d_reg()
    TestLearn.test_learn_3d_class()
    TestLearn.test_learn_2d_reg()
    TestLearn.test_transfer()
