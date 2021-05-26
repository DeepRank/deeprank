import os
import h5py
import tempfile
import shutil

from deeprank.features.ResidueContacts import __compute_feature__

test_path = os.path.dirname(os.path.realpath(__file__))

def test_compute_feature():
    pdb_path = os.path.join(test_path, '1AK4/atomic_features/1AK4_100w.pdb')

    tmp_path = tempfile.mkdtemp()
    try:

        with h5py.File(os.path.join(tmp_path, "test.hdf5"), 'w') as f:

            molgrp = f.require_group('1AK4')

            molgrp.require_group('features')
            molgrp.require_group('features_raw')

            __compute_feature__(pdb_path, molgrp['features'], molgrp['features_raw'], 'A', 25)

    finally:
        shutil.rmtree(tmp_path)
