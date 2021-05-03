from tempfile import mkdtemp
from shutil import rmtree
import os

from nose.tools import ok_
import h5py

from deeprank.models.mutant import PdbMutantSelection
from deeprank.features.accessibility import __compute_feature__, FEATURE_NAME


def test_compute_feature():

    tmp_dir_path = mkdtemp()
    try:
        hdf5_path = os.path.join(tmp_dir_path, "test.hdf5")
        with h5py.File(hdf5_path, 'w') as f5:
            feature_group = f5.require_group("features")
            raw_feature_group = f5.require_group("raw_features")
            mutant = PdbMutantSelection("test/101M.pdb", 'A', 25, 'C')

            __compute_feature__(mutant.pdb_path, feature_group, raw_feature_group, mutant)

            ok_(len(feature_group.get(FEATURE_NAME)) > 0)
    finally:
        rmtree(tmp_dir_path)
