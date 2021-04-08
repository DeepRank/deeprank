import os
import logging
import pkg_resources
from tempfile import mkdtemp
import shutil

import numpy
import h5py
from nose.tools import ok_, eq_

from deeprank.generate.GridTools import GridTools
from deeprank.models.mutant import PdbMutantSelection


_log = logging.getLogger(__name__)


def test_feature_mapping():

    pdb_name = "1XXX"

    mutant = PdbMutantSelection("1XXX.pdb", 'A', 1, 'V')

    # Build a temporary directory to store the test file.
    tmp_dir = mkdtemp()

    try:
        tmp_path = os.path.join(tmp_dir, "test.hdf5")

        with h5py.File(tmp_path, 'w') as f5:

            # Fill the HDF5 with data, before we give it to the grid.
            mol_group = f5.require_group(pdb_name)
            mol_group.attrs['type'] = 'molecule'

            atom_lines = [
                "ATOM      1  C   XXX A   1       0.000   0.000   0.000  1.00  0.00      C   C\n",
                "ATOM      2 CA   XXX A   2       1.000   1.000   1.000  1.00  0.00      C   C\n",
                "ATOM      3  N   XXX A   2      -1.000  -1.000  -1.000  1.00  0.00      C   N\n",
                "ATOM      4  N   XXX A   3      20.000  20.000  20.000  1.00  0.00      C   N\n",
            ]
            data = numpy.array(atom_lines).astype('|S78')
            mol_group.create_dataset('complex', data=data)

            feature_group = mol_group.require_group('features')

            feature_type_name = "testfeature"
            chain_id = "A"
            chain_number = 0
            position = [10.0, 10.0, 10.0]  # this should fit inside the grid
            value = 0.123

            data = numpy.array([[chain_number] + position + [value]])
            feature_group.create_dataset(feature_type_name, data=data)

            points_count = 30

            # Build the grid and map the features.
            gridtools = GridTools(mol_group, mutant,
                                  number_of_points=points_count, resolution=1.0,
                                  atomic_densities={'C': 1.7},  # only collect density data on carbon
                                  feature=[feature_type_name],
                                  try_sparse=False)

            # Check that the test feature and carbon atomic density are mapped to the grid.
            for feature_group_name in ["%s/mapped_features/AtomicDensities_ind" % pdb_name,
                                       "%s/mapped_features/Feature_ind" % pdb_name]:

                # Check that the feature group actually exists, before checking inside it.
                ok_(feature_group_name in f5)
                ok_(len(f5[feature_group_name]) > 0)

                for feature_name in f5[feature_group_name]:

                    _log.debug("check {}.{}".format(feature_group_name, feature_name))

                    # The number of values should match the grid settings.
                    eq_(f5[feature_group_name][feature_name].attrs['sparse'], False)
                    eq_(f5[feature_group_name][feature_name]['value'].shape, (points_count, points_count, points_count))

                    # It should be nonzero at least at one grid point.
                    ok_(numpy.sum(f5[feature_group_name][feature_name]['value']) > 0.0)
    finally:
        # Whether the test completes successfully or not, it needs to clean up after itself.
        shutil.rmtree(tmp_dir)
