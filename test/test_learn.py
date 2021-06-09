import os
from tempfile import mkdtemp
from shutil import rmtree

import numpy
import torch.optim as optim
from nose.tools import ok_

from deeprank.models.mutant import PdbMutantSelection
from deeprank.generate.DataGenerator import DataGenerator
from deeprank.learn.DataSet import DataSet
from deeprank.learn.NeuralNet import NeuralNet
from deeprank.learn.model3d import cnn_reg
import deeprank.config


deeprank.config.DEBUG = True


def test_learn():
    """ This test will simply run deeprank's learning code. It doesn't
        test any particular feature or target classes.

        The result of deeprank's learning is not verified. This test
        only runs the code to be sure there are no exceptions thrown.
    """

    feature_modules = ["test.feature.feature1", "test.feature.feature2"]
    target_modules = ["test.target.target1"]
    pdb_path = "test/101M.pdb"
    pssm_paths = {"A": "101M.A.pdb.pssm"}

    atomic_densities = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
    grid_info = {
       'number_of_points': [30,30,30],
       'resolution': [1.,1.,1.],
       'atomic_densities': atomic_densities,
    }

    mutant = PdbMutantSelection(pdb_path, "A", 10, "C", pssm_paths)

    work_dir_path = mkdtemp()
    try:
        hdf5_path = os.path.join(work_dir_path, "test.hdf5")

        # data_augmentation has been set to a high number, so that
        # the train, valid and test set can be large enough.
        data_generator = DataGenerator([mutant], data_augmentation=50,
                                       compute_targets=target_modules,
                                       compute_features=feature_modules,
                                       hdf5=hdf5_path)

        data_generator.create_database()

        data_generator.map_features(grid_info)

        dataset = DataSet(hdf5_path, grid_info=grid_info,
                          select_feature='all',
                          select_target='target1',
                          normalize_features=True,
                          dict_filter={'target1':'>=1'})

        ok_(len(dataset) > 0)
        ok_(dataset[0] is not None)

        net_output_dir_path = os.path.join(work_dir_path, 'net-output')
        neural_net = NeuralNet(dataset, cnn_reg, model_type='3d',task='reg', save_hitrate=False,
                               cuda=False, plot=True, outdir=net_output_dir_path)

        neural_net.optimizer = optim.SGD(neural_net.net.parameters(),
                                         lr=0.001,
                                         momentum=0.9,
                                         weight_decay=0.005)

        neural_net.train(nepoch = 50, divide_trainset=0.8, train_batch_size = 5, num_workers=0)
    finally:
        rmtree(work_dir_path)
