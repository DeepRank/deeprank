import os
import unittest
from time import time
import shutil
from tempfile import mkdtemp
import logging
import pkg_resources

import numpy
import h5py
from nose.tools import eq_, ok_

from deeprank.generate import *
from deeprank.models.mutant import PdbMutantSelection
from deeprank.tools.sparse import FLANgrid


_log = logging.getLogger(__name__)


def test_generate():
    """ This unit test checks that the HDF5 preprocessing code generates the correct data.
        It doesn't use any of the actual feature classes, but instead uses some simple test classes
        that have been created for this purpose.

        Data is expected to be mapped onto a 3D grid.
    """

    number_of_points = 30
    resolution = 1.0
    atomic_densities = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
    grid_info = {
        'number_of_points': [number_of_points, number_of_points, number_of_points],
        'resolution': [resolution, resolution, resolution],
        'atomic_densities': atomic_densities,
    }

    # These classes are made for testing, they give meaningless numbers.
    feature_names = ["test.feature.feature1", "test.feature.feature2"]
    target_names = ["test.target.target1"]

    pdb_path = os.path.join(pkg_resources.resource_filename("test", ''), "101m.pdb")
    mutants = [PdbMutantSelection(pdb_path, 'A', 25, 'A')]

    tmp_dir = mkdtemp()
    try:
        hdf5_path = os.path.join(tmp_dir, "data.hdf5")

        # Make the class put the data in the HDF5 file:
        data_generator = DataGenerator(mutants, None, target_names, feature_names, 1, hdf5_path)
        data_generator.create_database()
        data_generator.map_features(grid_info)

        # Read the resulting HDF5 file and check for all mutant data.
        with h5py.File(hdf5_path, 'r') as f5:
            eq_(list(f5.attrs['targets']), target_names)
            eq_(list(f5.attrs['features']), feature_names)

            for mutant in mutants:
                molecule_name = os.path.basename(os.path.splitext(mutant.pdb_path)[0])

                # Check that the right number of grid point coordinates have been stored and are equally away from each other:
                for coord in ['x', 'y', 'z']:
                    coords = f5["%s/grid_points/%s" % (molecule_name, coord)]
                    for i in range(number_of_points - 1):
                        eq_(coords[i + 1] - coords[i], resolution)

                # Check for mapped features in the HDF5 file:
                feature_path = "%s/mapped_features/Feature_ind" % molecule_name
                feature_keys = f5[feature_path].keys()
                for feature_name in feature_names:
                    feature_name = feature_name.split('.')[-1]
                    feature_key = [key for key in feature_keys if feature_name in key][0]
                    ok_(len(f5["%s/%s" % (feature_path, feature_key)]['value']) > 0)

                # Check that the atomic densities are present in the HDF5 file:
                for element_name in atomic_densities:
                    density_path = "%s/mapped_features/AtomicDensities_ind/%s_%s" % (molecule_name, element_name, mutant.chain_id)
                    ok_(len(f5[density_path]) > 0)

                # Check that the target values have been placed in the HDF5 file:
                for target_name in target_names:
                    target_name = target_name.split('.')[-1]
                    target_path = "%s/targets" % molecule_name
                    ok_(target_name in f5[target_path])
    finally:
        shutil.rmtree(tmp_dir)
