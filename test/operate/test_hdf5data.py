from tempfile import mkdtemp
from shutil import rmtree
import os

import h5py
from nose.tools import eq_

from deeprank.models.mutant import PdbMutantSelection
from deeprank.operate import hdf5data

def test_mutant():
    start_mutant = PdbMutantSelection("not/existent/pdb", 'A', 111, 'M', {'A': 'not/existent/pssm.A'})

    temp_dir_path = mkdtemp()
    try:
        hdf5_path = os.path.join(temp_dir_path, "test.hdf5")
        with h5py.File(hdf5_path, 'w') as f5:
            group = f5.require_group("mol1")

            hdf5data.store_mutant(group, start_mutant)

            end_mutant = hdf5data.load_mutant(group)

            eq_(start_mutant, end_mutant)
    finally:
        rmtree(temp_dir_path)
