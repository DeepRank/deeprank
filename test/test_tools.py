import unittest

from deeprank.tools import SASA


class TestTools(unittest.TestCase):
    """Test StructureSimialrity."""

    @staticmethod
    def test_sasa():
        """Test the SASA module."""

        # create the sql db
        pdb = './1AK4/decoys/1AK4_cm-it0_745.pdb'
        sasa = SASA(pdb)
        sasa.get_center(chain1='C', chain2='D')
        sasa.get_residue_center(chain1='C', chain2='D')
        sasa.neighbor_count(chain1='C', chain2='D')


if __name__ == '__main__':
    unittest.main()
