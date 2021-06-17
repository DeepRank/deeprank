from tempfile import mkdtemp
from shutil import rmtree
import os

from nose.tools import ok_
import h5py

from deeprank.models.variant import PdbVariantSelection
from deeprank.features.accessibility import __compute_feature__, FEATURE_NAME


def test_compute_feature():

    tmp_dir_path = mkdtemp()
    try:
        hdf5_path = os.path.join(tmp_dir_path, "test.hdf5")
        with h5py.File(hdf5_path, 'w') as f5:
            feature_group = f5.require_group("features")
            raw_feature_group = f5.require_group("raw_features")
            variant = PdbVariantSelection("test/101M.pdb", 'A', 25, 'C')

            __compute_feature__(variant.pdb_path, feature_group, raw_feature_group, variant)

            # Did the feature get stored:
            ok_(len(feature_group.get(FEATURE_NAME)) > 0)

            # There must be buried atoms:
            ok_(any([key_value[-1] == 0.0 for key_value in feature_group.get(FEATURE_NAME)]))

            # There must be accessible atoms:
            ok_(any([key_value[-1] > 0.0 for key_value in feature_group.get(FEATURE_NAME)]))
    finally:
        rmtree(tmp_dir_path)
