from deeprank.generate import *
import os
from time import time


"""Test the data generation process."""

h5file = './1ak4.hdf5'
pdb_source     = './decoys_pdbFLs/1AK4/water/'
pdb_native     = './bound_pdb/'


database = DataGenerator(pdb_source=pdb_src,pdb_native=pdb_native,
                         data_augmentation = 0,
                         compute_targets  = ['deeprank.targets.dockQ','deeprank.targets.binary_class'],
                         compute_features = ['deeprank.features.AtomicFeature',
                                             'deeprank.features.NaivePSSM',
                                             'deeprank.features.PSSM_IC',
                                             'deeprank.features.BSA',
                                             'deeprank.features.FullPSSM',
                                             'deeprank.features.ResidueDensity'],
                         hdf5=h5file)

#create new files
print('{:25s}'.format('Create new database') + database.hdf5)
database.create_database(prog_bar=True)
print(' '*25 + '--> Done in %f s.' %(time()-t0))

