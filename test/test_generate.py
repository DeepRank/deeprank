import unittest
from deeprank.generate import *
import os
from time import time

class TestGenerateData(unittest.TestCase):
    """Test the data generation process."""

    h5file = './1ak4.hdf5'
    pdb_source     = ['./1AK4/decoys/']
    pdb_native     = ['./1AK4/native/']

    def test_1_generate(self):
        """Generate the database."""

        # clean old files
        files = ['1ak4.hdf5','1ak4_norm.pckl']
        for f in files:
            if os.path.isfile(f):
                os.remove(f)

        #init the data assembler
        database = DataGenerator(pdb_source=self.pdb_source,pdb_native=self.pdb_native,
                                 compute_targets  = ['deeprank.targets.dockQ'],
                                 compute_features = ['deeprank.features.AtomicFeature',
                                                     'deeprank.features.NaivePSSM',
                                                     'deeprank.features.PSSM_IC',
                                                     'deeprank.features.BSA'],
                                 hdf5=self.h5file)

        #create new files
        if not os.path.isfile(database.hdf5):
            t0 = time()
            print('{:25s}'.format('Create new database') + database.hdf5)
            database.create_database(prog_bar=True)
            print(' '*25 + '--> Done in %f s.' %(time()-t0))
        else:
            print('{:25s}'.format('Use existing database') + database.hdf5)

        # map the features
        grid_info = {
            'number_of_points' : [30,30,30],
            'resolution' : [1.,1.,1.],
            'atomic_densities' : {'CA':3.5,'N':3.5,'O':3.5,'C':3.5},
        }

        t0 = time()
        print('{:25s}'.format('Map features in database') + database.hdf5)
        database.map_features(grid_info,try_sparse=True,time=False,prog_bar=True)
        print(' '*25 + '--> Done in %f s.' %(time()-t0))

        # get the normalization
        t0 =time()
        print('{:25s}'.format('Normalization') + database.hdf5)
        norm = NormalizeData(self.h5file)
        norm.get()
        print(' '*25 + '--> Done in %f s.' %(time()-t0))


    def test_2_add_target(self):
        """Add a target to the database."""

        #init the data assembler
        database = DataGenerator(pdb_source=None,pdb_native=None,data_augmentation=None,
                                 compute_targets  = ['deeprank.targets.binary_class'], hdf5=self.h5file)

        t0 = time()
        print('{:25s}'.format('Add new target in database') + database.hdf5)
        database.add_target(prog_bar=True)
        print(' '*25 + '--> Done in %f s.' %(time()-t0))

    def test_3_add_feature(self):
        """Add a feature to the database."""

        #init the data assembler
        database = DataGenerator(pdb_source=None,pdb_native=None,data_augmentation=None,
                                 compute_features  = ['deeprank.features.ResidueDensity'], hdf5=self.h5file)

        t0 =time()
        print('{:25s}'.format('Add new feature in database') + database.hdf5)
        database.add_feature(prog_bar=True)
        print(' '*25 + '--> Done in %f s.' %(time()-t0))

        t0 = time()
        print('{:25s}'.format('Map new feature in database') + database.hdf5)
        database.map_features(try_sparse=True,time=False,prog_bar=True)
        print(' '*25 + '--> Done in %f s.' %(time()-t0))

if __name__ == "__main__":
    unittest.main()


