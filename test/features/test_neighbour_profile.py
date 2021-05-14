import os
from tempfile import mkdtemp
from shutil import rmtree

import h5py
from nose.tools import ok_

from deeprank.features.neighbour_profile import (__compute_feature__,
                                                 IC_FEATURE_NAME,
                                                 get_probability_feature_name)
from deeprank.models.mutant import PdbMutantSelection


def test_feature():
    tmp_dir_path = mkdtemp()

    try:
        hdf5_path = os.path.join(tmp_dir_path, 'test.hdf5')

        mutant = PdbMutantSelection("test/101M.pdb", "A", 25, "W",
                                    {'A': "test/101M.A.pdb.pssm"})

        with h5py.File(hdf5_path, 'w') as f5:
            group = f5.require_group("features")
            __compute_feature__(mutant.pdb_path, group, None, mutant)

            # Check that the features are present on the grid:
            ok_(len(group.get(get_probability_feature_name("ALA"))) > 0)
            ok_(len(group.get(IC_FEATURE_NAME)) > 0)
    finally:
        rmtree(tmp_dir_path)

