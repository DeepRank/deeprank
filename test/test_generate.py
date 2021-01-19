import os
import unittest
from time import time
import shutil
from deeprank.generate import *


"""
Some requirement of the naming of the files:
    1. case ID canNOT have underscore '_', e.g., '1ACB_CD'
    2. decoy file name should have this format: 2w83-AB_20.pdb (caseID_xxx.pdb)
    3. pssm file name should have this format: 2w83-AB.A.pssm (caseID.chainID.pssm or caseID.chainID.pdb.pssm)
"""


class TestGenerateData(unittest.TestCase):
    """Test the data generation process."""

    h5file = ['./1ak4.hdf5', 'native.hdf5']
    pdb_source = ['./1AK4/decoys/', './1AK4/native/']
    # pdb_native is only used to calculate i-RMSD, dockQ and so on. The native
    # pdb files will not be saved in the hdf5 file
    pdb_native = ['./1AK4/native/']

    def test_1_generate(self):
        """Generate the database."""

        # clean old files
        files = [
            '1ak4.hdf5',
            '1ak4_norm.pckl',
            'native.hdf5',
            'native_norm.pckl']
        for f in files:
            if os.path.isfile(f):
                os.remove(f)

        # init the data assembler
        for h5, src in zip(self.h5file, self.pdb_source):

            database = DataGenerator(
                chain1='C',
                chain2='D',
                pdb_source=src,
                pdb_native=self.pdb_native,
                pssm_source='./1AK4/pssm_new/',
                data_augmentation=1,
                compute_targets=[
                    'deeprank.targets.dockQ',
                    'deeprank.targets.binary_class',
                    'deeprank.targets.capri_class'],
                compute_features=[
                    'deeprank.features.AtomicFeature',
                    'deeprank.features.FullPSSM',
                    'deeprank.features.PSSM_IC',
                    'deeprank.features.BSA',
                    'deeprank.features.ResidueDensity'],
                hdf5=h5)

            # create new files
            if not os.path.isfile(database.hdf5):
                t0 = time()
                print('{:25s}'.format('Create new database') + database.hdf5)
                database.create_database(prog_bar=True, random_seed=2019)
                print(' ' * 25 + '--> Done in %f s.' % (time() - t0))
            else:
                print('{:25s}'.format('Use existing database') + database.hdf5)

            # map the features
            grid_info = {
                'number_of_points': [10, 10, 10],
                'resolution': [3., 3., 3.],
                'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8},
            }

            t0 = time()
            print('{:25s}'.format('Map features in database') + database.hdf5)
            database.map_features(
                grid_info,
                try_sparse=True,
                time=False,
                prog_bar=True)
            print(' ' * 25 + '--> Done in %f s.' % (time() - t0))

            # get the normalization
            t0 = time()
            print('{:25s}'.format('Normalization') + database.hdf5)
            norm = NormalizeData(h5)
            norm.get()
            print(' ' * 25 + '--> Done in %f s.' % (time() - t0))

    def test_1_generate_mapfly(self):
        """Generate the database."""

        # clean old files
        files = [
            '1ak4_mapfly.hdf5',
            '1ak4_mapfly.pckl'
            ]
        for f in files:
            if os.path.isfile(f):
                os.remove(f)

        h5 = "./1ak4_mapfly.hdf5"
        src = self.pdb_source[0]

        # init the data assembler

        database = DataGenerator(
            chain1='C',
            chain2='D',
            pdb_source=src,
            pdb_native=self.pdb_native,
            pssm_source='./1AK4/pssm_new/',
            # data_augmentation=1,
            compute_targets=[
                'deeprank.targets.dockQ',
                'deeprank.targets.binary_class',
                'deeprank.targets.capri_class'],
            compute_features=[
                'deeprank.features.AtomicFeature',
                'deeprank.features.FullPSSM',
                'deeprank.features.PSSM_IC',
                'deeprank.features.BSA',
                'deeprank.features.ResidueDensity'],
            hdf5=h5)

        # create new files
        print('{:25s}'.format('Create new database') + database.hdf5)
        database.create_database(prog_bar=True)

    def test_2_add_target(self):
        """Add a target (e.g., class labels) to the database."""

        for h5 in self.h5file:

            # init the data assembler
            database = DataGenerator(chain1='C', chain2='D',
                compute_targets=['deeprank.targets.binary_class'], hdf5=h5)

            t0 = time()
            print(
                '{:25s}'.format('Add new target in database') +
                database.hdf5)
            database.add_target(prog_bar=True)
            print(' ' * 25 + '--> Done in %f s.' % (time() - t0))

    def test_3_add_unique_target(self):
        """"Add a unique target to all the confs."""
        for h5 in self.h5file:
            database = DataGenerator(chain1='C', chain2='D', hdf5=h5)
            database.add_unique_target({'XX': 1.0})

    def test_4_add_feature(self):
        """Add a feature to the database."""

        for h5 in self.h5file:

            # init the data assembler
            database = DataGenerator(
                chain1='C',
                chain2='D',
                pdb_source=None,
                pdb_native=None,
                data_augmentation=None,
                pssm_source='./1AK4/pssm_new/',
                compute_features=['deeprank.features.FullPSSM'],
                hdf5=h5)

            t0 = time()
            print(
                '{:25s}'.format('Add new feature in database') +
                database.hdf5)
            database.add_feature(prog_bar=True)
            print(' ' * 25 + '--> Done in %f s.' % (time() - t0))

            t0 = time()
            print(
                '{:25s}'.format('Map new feature in database') +
                database.hdf5)
            database.map_features(try_sparse=True, time=False, prog_bar=True)
            print(' ' * 25 + '--> Done in %f s.' % (time() - t0))

            # get the normalization
            t0 = time()
            print('{:25s}'.format('Normalization') + database.hdf5)
            norm = NormalizeData(h5)
            norm.get()
            print(' ' * 25 + '--> Done in %f s.' % (time() - t0))


    def test_5_align(self):
        """create a database where all the complex are aligned in the z direction."""

        # clean old files
        files = [
            '1ak4_aligned.hdf5',
            '1ak4_aligned_norm.pckl']

        for f in files:
            if os.path.isfile(f):
                os.remove(f)

        database = DataGenerator(
            chain1='C',
            chain2='D',
            pdb_source='./1AK4/decoys/',
            pdb_native=self.pdb_native,
            pssm_source='./1AK4/pssm_new/',
            align={"axis":'z'},
            data_augmentation=1,
            compute_targets=['deeprank.targets.dockQ'],
            compute_features=['deeprank.features.AtomicFeature'],
            hdf5='./1ak4_aligned.hdf5')

        # create the database
        if not os.path.isfile(database.hdf5):
            t0 = time()
            print('{:25s}'.format('Create new database') + database.hdf5)
            database.create_database(prog_bar=True, random_seed=2019)
            print(' ' * 25 + '--> Done in %f s.' % (time() - t0))
        else:
            print('{:25s}'.format('Use existing database') + database.hdf5)

    def test_6_align_interface(self):
        """create a database where all the interface are aligned in the xy plane."""

        # clean old files
        files = [
            '1ak4_aligned_interface.hdf5',
            '1ak4_aligned_interface_norm.pckl']

        for f in files:
            if os.path.isfile(f):
                os.remove(f)

        database = DataGenerator(
            pdb_source='./1AK4/decoys/',
            pdb_native=self.pdb_native,
            pssm_source='./1AK4/pssm_new/',
            align={"plane":'xy', "selection":'interface'},
            data_augmentation=1,
            compute_targets=['deeprank.targets.dockQ'],
            compute_features=['deeprank.features.AtomicFeature'],
            hdf5='./1ak4_aligned_interface.hdf5',
            chain1='C',
            chain2='D')

        # create the database
        if not os.path.isfile(database.hdf5):
            t0 = time()
            print('{:25s}'.format('Create new database') + database.hdf5)
            database.create_database(prog_bar=True, random_seed=2019)
            print(' ' * 25 + '--> Done in %f s.' % (time() - t0))
        else:
            print('{:25s}'.format('Use existing database') + database.hdf5)

    def test_7_realign(self):
        '''Realign existing pdbs.'''

        src_name = './1ak4.hdf5'
        copy_name = './1ak4_aligned.hdf5'

        os.remove(copy_name)
        shutil.copy(src_name,copy_name)

        database = DataGenerator(hdf5=copy_name, chain1='C', chain2='D')
        database.realign_complexes(align={'axis':'z'})

    def test_8_aug_data(self):

        src_name = './1ak4.hdf5'
        copy_name = './1ak4_aug.hdf5'

        shutil.copy(src_name, copy_name)

        database = DataGenerator(hdf5=copy_name, chain1='C', chain2='D')
        database.aug_data(augmentation=2, keep_existing_aug=True)
        grid_info = {
            'number_of_points': [10, 10, 10],
            'resolution': [3., 3., 3.],
            'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8},
            'feature': ['PSSM_ALA','RCD_total', 'bsa', 'charge',],
        }
        database.map_features(
            grid_info,
            try_sparse=True,
            time=False,
            prog_bar=False,
        )

if __name__ == "__main__":

    # unittest.main()
    inst = TestGenerateData()
    inst.test_1_generate()
    inst.test_1_generate_mapfly()
    inst.test_3_add_unique_target()
    inst.test_4_add_feature()
    inst.test_5_align()
    inst.test_6_align_interface()
    inst.test_7_realign()
    inst.test_8_aug_data()