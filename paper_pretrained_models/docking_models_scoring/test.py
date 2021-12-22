from mpi4py import MPI
import sys
import os
import re
import glob
from time import time

from deeprank.generate import *
from deeprank.learn import NeuralNet
from model_280619 import cnn_class

comm = MPI.COMM_WORLD
pdb_source = '../../test/1AK4/decoys/'
pssm_source = '../../test/1AK4/pssm_new/'

database = DataGenerator(pdb_source= pdb_source, #path to the models  
                         pssm_source= pssm_source, #path to the pssm data
                         data_augmentation = None,
                         chain1='C', chain2='D',
                         compute_features = ['deeprank.features.AtomicFeature', 'deeprank.features.FullPSSM',
                         'deeprank.features.PSSM_IC', 'deeprank.features.BSA', 'deeprank.features.ResidueDensity'],
                         hdf5='output.hdf5',mpi_comm=comm)


# compute features/targets, and write to hdf5 file
print('{:25s}'.format('Create new database') + database.hdf5)
database.create_database(prog_bar=True)

# define the 3D grid
grid_info = {
'number_of_points' : [30,30,30],
'resolution' : [1.,1.,1.],
'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8},
}

# map the features to the 3D grid
print('{:25s}'.format('Map features in database') + database.hdf5)
database.map_features(grid_info,try_sparse=True, time=False, prog_bar=True)

# select the pre-trained model
model_data = 'best_train_model.pt'
database = glob.glob('*.hdf5')

model = NeuralNet(database, cnn_class,
                  pretrained_model=model_data, save_hitrate=False)

# test the pre-trained model on new data
model.test()
