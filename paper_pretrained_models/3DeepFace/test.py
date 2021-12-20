"""
Test 3DeepFace models

"""
import os
import sys
import glob
from deeprank.learn import *

# to set your own architecture
from arch_001_02 import cnn_class as cnn3d_class

database = DataGenerator(pdb_source=pdb_source,
                         chain1='A', chain2='B',
                         pssm_source=pssm_source,
                         compute_features = ['deeprank.features.AtomicFeature',
                                             'deeprank.features.FullPSSM',
                                             'deeprank.features.PSSM_IC',
                                             'deeprank.features.BSA',
                                             'deeprank.features.ResidueDensity'],
                         data_augmentation = 30, # rotate complexes
                         hdf5=./)

# compute features/targets, and write to hdf5 file
print('{:25s}'.format('Create new database') + database.hdf5)
database.create_database()

# map the features
grid_info = {
    'number_of_points': [10, 10, 10],
    'resolution': [3., 3., 3.],
    'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8},
}

# map the features to the 3D grid
print('{:25s}'.format('Map features in database') + database.hdf5)
database.map_features(grid_info, try_sparse=True, time=False, prog_bar=True)

# You need to add path for the dataset
database = glob.glob('*hdf5')

model = NeuralNet(database,cnn3d_class,pretrained_model='best_model.pt', outdir=outpath)

# test the pre-trained model on new data
model.test()
