import os
import h5py
import tempfile
import shutil

from deeprank.features.atomic_contacts import __compute_feature__
from deeprank.models.mutant import PdbMutantSelection


def test_compute_feature():
    pdb_path = "test/1AK4/atomic_features/1AK4_100w.pdb"

    mutant = PdbMutantSelection(pdb_path, 'C', 25, 'A')

    tmp_path = tempfile.mkdtemp()
    try:

        with h5py.File(os.path.join(tmp_path, "test.hdf5"), 'w') as f:

            molgrp = f.require_group('1AK4')

            molgrp.require_group('features')
            molgrp.require_group('features_raw')

            __compute_feature__(pdb_path, molgrp['features'], molgrp['features_raw'], mutant)

    finally:
        shutil.rmtree(tmp_path)
