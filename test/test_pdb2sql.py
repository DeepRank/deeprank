import unittest
import numpy as np
from deeprank.tools import pdb2sql

class TestPDB2SQL(unittest.TestCase):
    """Test PDB2SQL."""

    def test_read(self):
        """Read a pdb and create a sql db."""

        #db.prettyprint()
        self.db.get_colnames()
        self.db.exportpdb('chainA.pdb', chainID = 'A')

    def test_get(self):
        """Test get with large number of index."""

        index = list(range(1200))
        self.db.get('x,y,z', rowID = index)

    @unittest.expectedFailure
    def test_get_fails(self):
        """Test get with a too large number of conds."""

        index_res = list(range(100))
        index_atoms = list(range(1200))
        self.db.get('x,y,z', resSeq = index_res, rowID = index_atoms)

    def test_add_column(self):
        """Add a new column to the db and change its values."""

        self.db.add_column('CHARGE', 'FLOAT')
        self.db.put('CHARGE', 0.1)
        n = 100
        q = np.random.rand(n)
        ind = list(range(n))
        self.db.update_column('CHARGE', q, index = ind)

    def test_update(self):
        """Update the database."""

        n = 200
        index = list(range(n))
        vals = np.random.rand(n, 3)
        self.db.update('x,y,z',vals, rowID = index)
        self.db.update_xyz(vals, index = index)

    def test_manip(self):
        """Manipualte part of the protein."""

        vect = np.random.rand(3)
        self.db.translation(vect, chainID = 'A')

        axis = np.random.rand(3)
        angle = np.random.rand()
        self.db.rotation_around_axis(axis, angle, chainID = 'B')

        a,b,c = np.random.rand(3)
        self.db.rotation_euler(a,b,c,resName='VAL')

        mat = np.random.rand(3,3)
        self.db.rotation_matrix(mat,chainID='A')

    def setUp(self):
        mol = './1AK4/decoys/1AK4_1w.pdb'
        self.db = pdb2sql(mol)

    def tearDown(self):
        self.db.close()

if __name__ == '__main__':
    unittest.main()