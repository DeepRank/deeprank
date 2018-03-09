import numpy as np
import unittest
from deeprank.tools import pdb2sql
from deeprank.tools import SASA

class TestTools(unittest.TestCase):
    """Test StructureSimialrity."""

    @staticmethod
    def test_pdb2sql():

        # create the sql db
        pdb = './1AK4/decoys/1AK4_1w.pdb'
        db = pdb2sql(pdb)

        # fix chain name
        db._fix_chainID()

        # get column name
        db.get_colnames()

        # print
        db.prettyprint()
        db.uglyprint()

    @staticmethod
    def test_sasa():

        # create the sql db
        pdb = './1AK4/decoys/1AK4_1w.pdb'
        sasa = SASA(pdb)
        sasa.get_center()
        sasa.get_residue_center()
        #sasa.neighbor_count()

if __name__ == '__main__':
    unittest.main()