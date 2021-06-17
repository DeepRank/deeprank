import os
from tempfile import mkdtemp
from shutil import rmtree

import h5py
from nose.tools import ok_, eq_

from deeprank.models.mutant import PdbMutantSelection, MutantClass
from deeprank.targets.mutant_class import __compute_target__


def test_annotate():
    pdb_path = "test/101M.pdb"
    pssm_paths = {"A": "101M.A.pdb.pssm"}
    mutant = PdbMutantSelection(pdb_path, "A", 10, "C", pssm_paths, mutant_class=MutantClass.BENIGN)

    work_dir_path = mkdtemp()
    try:
        hdf5_path = os.path.join(work_dir_path, "test.hdf5")
        with h5py.File(hdf5_path, 'w') as f5:
            target_group = f5.require_group("targets")
            __compute_target__(mutant, target_group)

            ok_("class" in target_group)
            ok_(target_group["class"], 0.0)
    finally:
        rmtree(work_dir_path)
