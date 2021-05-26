import os
import h5py
import tempfile
import shutil

import numpy
from nose.tools import ok_, eq_

from deeprank.features.atomic_contacts import __compute_feature__
from deeprank.models.mutant import PdbMutantSelection


def test_compute_feature():
    pdb_path = "test/1AK4/native/1AK4.pdb"

    mutant = PdbMutantSelection(pdb_path, 'C', 25, 'A')

    tmp_path = tempfile.mkdtemp()
    try:
        with h5py.File(os.path.join(tmp_path, "test.hdf5"), 'w') as f:

            molgrp = f.require_group('1AK4')

            features_group = molgrp.require_group('features')
            raw_group = molgrp.require_group('features_raw')

            __compute_feature__(pdb_path, features_group, raw_group, mutant)

            vdwaals_data = features_group['vdwaals']
            coulomb_data = features_group['coulomb']
            charge_data = features_group['charge']

            # Expected: x, y, z, value (=4)
            ok_(vdwaals_data.size > 0)
            ok_(vdwaals_data.size % 4 == 0)
            ok_(coulomb_data.size > 0)
            ok_(coulomb_data.size % 4 == 0)
            ok_(charge_data.size > 0)
            ok_(charge_data.size % 4 == 0)

            vdwaals_data_raw = raw_group['vdwaals_raw']
            coulomb_data_raw = raw_group['coulomb_raw']
            charge_data_raw = raw_group['charge_raw']

            ok_(vdwaals_data_raw.size > 0)
            eq_(type(vdwaals_data_raw[0]), numpy.bytes_)
            ok_(coulomb_data_raw.size > 0)
            eq_(type(coulomb_data_raw[0]), numpy.bytes_)
            ok_(charge_data_raw.size > 0)
            eq_(type(charge_data_raw[0]), numpy.bytes_)

    finally:
        shutil.rmtree(tmp_path)
