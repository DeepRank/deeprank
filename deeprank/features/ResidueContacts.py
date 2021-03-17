import logging
import pdb2sql
from deeprank.feature.FeatureClass import FeatureClass


_log = logging.getLogger(__name__)


class ResidueContacts(FeatureClass):
    RESIDUE_KEY = ["chainID", "resSeq", "resName"]
    ATOM_KEY = ["chainID", "resSeq", "resName", "name"]
    EPS0 = 1.0
    C = 332.0636

    def __init__(self, pdb_path, chain_id, residue_number):
        super.__init__("Atomic")

        self.pdb_path = pdb_path
        self.chain_id = chain_id
        self.residue_number = residue_number

    def __enter__(self):
        self.pdbsql = pdb2sql.interface(self.pdb_path)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.pdbsql._close()

    def evaluate(self):
        self._read_charges()
        self._read_vdw()
        self._read_patch()
        self._assign_parameters()

        self._evaluate_vdw()
        self._evaluate_coulomb()
        self._evaluate_charges()



def __compute_feature__(pdb_path, featgrp, featgrp_raw, chain_id, residue_number):
    with ResidueContacts(pdb_path, chain_id, residue_number) as feature_object:

        feature_object.evaluate()

        # export in the hdf5 file
        atfeat.export_dataxyz_hdf5(featgrp)
        atfeat.export_data_hdf5(featgrp_raw)


