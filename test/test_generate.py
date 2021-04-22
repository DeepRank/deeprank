import os
import unittest
from time import time
import shutil
from tempfile import mkdtemp

import numpy
import h5py
from nose.tools import eq_, ok_

from deeprank.generate import *
from deeprank.models.mutant import PdbMutantSelection


def test_generate():

    number_of_points = 30
    atomic_densities = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
    grid_info = {
        'number_of_points': [number_of_points, number_of_points, number_of_points],
        'resolution': [1.,1.,1.],
        'atomic_densities': atomic_densities,
    }

    feature_names = ["test.feature1", "test.feature2"]
    target_names = ["test.target1"]

    mutants = [PdbMutantSelection("test/101m.pdb", 'A', 25, 'A')]

    tmp_dir = mkdtemp()
    try:
        hdf5_path = os.path.join(tmp_dir, "data.hdf5")

        data_generator = DataGenerator(mutants, target_names, feature_names, 1, hdf5_path)
        data_generator.create_database()
        data_generator.map_features(grid_info)

        with h5py.File(hdf5_path, 'r') as f5:
            eq_(f5.attrs['targets'], target_names)
            eq_(f5.attrs['features'], feature_names)

            for mutant in mutants:
                molecule_name = os.path.basename(os.path.splitext(mutant.pdb_path)[0])

                for feature_name in feature_names:
                    feature_path = "%s/mapped_features/Feature_ind/%s" % (molecule_name, feature_name)
                    feature_data = f5[feature_path]
                    eq_(numpy.shape(feature_data), (number_of_points, number_of_points, number_of_points))

                for element_name in grid_info['atomic_densities']:
                    density_path = "%s/mapped_features/AtomicDensities_ind/%s" % (molecule_name, element_name)
                    density_data = f5[density_path]
                    eq_(numpy.shape(density_data), (number_of_points, number_of_points, number_of_points))

                for target_name in target_names:
                    target_path = "%s/targets/%s"
                    target_data = f5[target_path]
                    ok_(len(target_data) > 0)
    finally:
        shutil.rmtree(tmp_dir)
