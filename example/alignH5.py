# This script is used to align complexes in hdf5.
# INPUT:
#   1. a hdf5 file that contains pdb of all complexes (it is the output of generate_dataset_noalign.py).
#   2. pssm files if the pssm feature is needed for the training later


from deeprank.generate import *
from mpi4py import MPI

comm = MPI.COMM_WORLD

# name of the hdf5 to align
h5file = './hdf5/1ak4.hdf5'

# where to find the pssm
pssm_source = '../test/1AK4/pssm_new/'


# align the principle component 1 of complexes to axis z
newdb = DataGenerator(hdf5=h5file)
newdb.realign_complexes(align={'axis':'z'})


# define the 3D grid
# grid_info = {
#   'number_of_points': [30,30,30],
#   'resolution': [1.,1.,1.],
#   'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8},
# }

# generate the grid
#print('{:25s}'.format('Generate the grid') + database.hdf5)
#database.precompute_grid(grid_info,try_sparse=True, time=False, prog_bar=True)


# print('{:25s}'.format('Map features in database') + database.hdf5)
# database.map_features(grid_info,try_sparse=True, time=False, prog_bar=True)

# # get the normalization of the features
# print('{:25s}'.format('Normalization') + database.hdf5)
# norm = NormalizeData(database.hdf5)
# norm.get()
