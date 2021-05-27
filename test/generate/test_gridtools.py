import os
import pkg_resources
from tempfile import mkdtemp
import shutil

import numpy
import h5py
from nose.tools import ok_, eq_

from deeprank.generate.GridTools import GridTools
from deeprank.models.mutant import PdbMutantSelection


def _get_feature_grid(hdf5, feature_group_name, feature_name, points_count):

    # Check that the feature grid exists:
    assert feature_group_name in hdf5, \
        "{} not in hdf5, candidates are: {}".format(",".join(hdf5.keys()))
    assert feature_name in hdf5[feature_group_name], \
        "{} not in feature group, candidates are: {}".format(feature_name, ",".join(hdf5[feature_group_name].keys()))

    # Check the grid size
    eq_(hdf5[feature_group_name][feature_name].attrs['sparse'], False)
    eq_(hdf5[feature_group_name][feature_name]['value'].shape, (points_count, points_count, points_count))

    return hdf5[feature_group_name][feature_name]['value']


def gt_(value1, value2):
    assert value1 > value2, "{} <= {}".format(value1, value2)


def lt_(value1, value2):
    assert value1 < value2, "{} >= {}".format(value1, value2)


def test_feature_mapping():
    """ In this test, we investigate a set of five atoms. We make the grid tool take
        the atoms in a 20 A radius around the first atom and compute the carbon density grid around it.
        We're also positioning a feature on a grid and investigate its resulting distribution.

        Grid values should be high close to the set position and low everywhere else.
    """

    pdb_name = "1XXX"


    # Build a temporary directory to store the test file.
    tmp_dir = mkdtemp()

    try:
        pdb_path = os.path.join(tmp_dir, "%s" % pdb_name)

        with open(pdb_path, 'wt') as f:
            for line in [
                "ATOM      1  C   XXX A   1       0.000   0.000   0.000  1.00  0.00      C   C\n",
                "ATOM      2 CA   XXX A   2       1.000   1.000   1.000  1.00  0.00      C   C\n",
                "ATOM      3  N   XXX A   2      -1.000  -1.000  -1.000  1.00  0.00      C   N\n",
                "ATOM      4  N   XXX A   3      10.000  10.000  10.000  1.00  0.00      C   N\n",
                "ATOM      5  C   XXX A   4     -10.000 -10.000 -10.000  1.00  0.00      C   C\n",
            ]:
               f.write(line) 

        mutant = PdbMutantSelection(pdb_path, 'A', 1, 'V')

        tmp_path = os.path.join(tmp_dir, "test.hdf5")

        with h5py.File(tmp_path, 'w') as f5:

            # Fill the HDF5 with data, before we give it to the grid.
            mol_group = f5.require_group(pdb_name)
            mol_group.attrs['type'] = 'molecule'
            mol_group.attrs['pdb_path'] = pdb_path

            feature_group = mol_group.require_group('features')

            feature_type_name = "testfeature"
            chain_id = "A"
            chain_number = 0
            position = [10.0, 10.0, 10.0]  # this should fit inside the grid
            value = 0.923

            data = numpy.array([[chain_number] + position + [value]])
            feature_group.create_dataset(feature_type_name, data=data)

            points_count = 30

            # Build the grid and map the features.
            gridtools = GridTools(mol_group, mutant,
                                  number_of_points=points_count, resolution=1.0,
                                  atomic_densities={'C': 1.7},  # only collect density data on carbon
                                  feature=[feature_type_name],
                                  contact_distance=20.0,  # just take all the atoms, close and far
                                  try_sparse=False)

            carbon_density_grid = _get_feature_grid(f5,
                                                    "%s/mapped_features/AtomicDensities_ind" % pdb_name,
                                                    "C_%s" % chain_id,
                                                    points_count)

            # Check that the gaussian is high at the carbon atom positions:
            gt_(carbon_density_grid[14][14][14], 0.1)
            gt_(carbon_density_grid[4][4][4], 0.1)

            # Check that nitrogens are not participating:
            lt_(carbon_density_grid[24][24][24], 0.001)

            feature_grid = _get_feature_grid(f5,
                                             "%s/mapped_features/Feature_ind" % pdb_name,
                                             "%s_chain000" % feature_type_name,
                                             points_count)

            # Check that the feature is high at the set position:
            gt_(feature_grid[24][24][24], 0.1)

            # Check that the feature is low where it was not set:
            lt_(feature_grid[0][0][0], 0.001)

    finally:
        # Whether the test completes successfully or not, it needs to clean up after itself.
        shutil.rmtree(tmp_dir)
