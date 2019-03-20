import unittest
from deeprank.tools import pdb2sql
from deeprank.tools import SASA


class TestTools(unittest.TestCase):
    """Test StructureSimialrity."""

    @staticmethod
    def test_pdb2sql():
        """Test the db2sql module."""

        # create the sql db
        pdb = './1AK4/decoys/1AK4_cm-it0_745.pdb'
        db = pdb2sql(pdb)
        db._fix_chainID()

        # get column name
        db.get_colnames()

        # print
        db.prettyprint()
        db.uglyprint()

    @staticmethod
    def test_sasa():
        """Test the SASA module."""

        # create the sql db
        pdb = './1AK4/decoys/1AK4_cm-it0_745.pdb'
        sasa = SASA(pdb)
        sasa.get_center()
        sasa.get_residue_center()
        sasa.neighbor_count()

if __name__ == '__main__':
    unittest.main()
