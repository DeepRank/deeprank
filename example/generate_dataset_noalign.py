from deeprank.generate import *
from mpi4py import MPI

comm = MPI.COMM_WORLD

# name of the hdf5 to generate
h5file = './hdf5/1ak4.hdf5'

# for each hdf5 file where to find the pdbs
pdb_source = '../test/1AK4/decoys/'

# where to find the native conformations
# pdb_native is only used to calculate i-RMSD, dockQ and so on.
# The native pdb files will not be saved in the hdf5 file
pdb_native = '../test/1AK4/native/'


# where to find the pssm
pssm_source = '../test/1AK4/pssm_new/'

# initialize the database
database = DataGenerator(
    pdb_source=pdb_source,
    pdb_native=pdb_native,
    pssm_source=pssm_source,
    data_augmentation=2,
    compute_targets=[
        'deeprank.targets.dockQ',
        'deeprank.targets.binary_class'],
    compute_features=[
        'deeprank.features.AtomicFeature',
        'deeprank.features.FullPSSM',
        'deeprank.features.PSSM_IC',
        'deeprank.features.BSA',
        'deeprank.features.ResidueDensity'],
    hdf5=h5file,
    mpi_comm=comm)


# create the database
# compute features/targets for all complexes
print('{:25s}'.format('Create new database') + database.hdf5)
database.create_database(prog_bar=True)


#define the 3D grid
grid_info = {
  'number_of_points' : [30,30,30],
  'resolution' : [1.,1.,1.],
  'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8},
}

# generate the grid and write the grid coordinates to hdf5
# This step can be skipped and is covered by map_features() below.
#
# print('{:25s}'.format('Generate the grid') + database.hdf5)
# database.precompute_grid(grid_info,try_sparse=True, time=False, prog_bar=True)

# map features to the 3D grid
print('{:25s}'.format('Map features in database') + database.hdf5)
database.map_features(grid_info,try_sparse=True, time=False, prog_bar=True)

# get the normalization of the features.
# This step can also be done the DataSet class (see learn.py)
#
# print('{:25s}'.format('Normalization') + database.hdf5)
# norm = NormalizeData(database.hdf5)
# norm.get()
