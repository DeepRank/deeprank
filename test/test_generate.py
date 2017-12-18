import sys
import deeprank.generate
import os
from time import time
def test_generate():

	

	# sources to assemble the data base
	pdb_source     = ['./1AK4/decoys/']
	pdb_native     = ['./1AK4/native/']

	#init the data assembler 
	database = deeprank.generate.DataGenerator(pdb_source=pdb_source,pdb_native=pdb_native,data_augmentation=None,
		                                       compute_targets  = ['deeprank.tools.targets.dockQ'],
		                                       compute_features = ['deeprank.tools.features.atomic'],
		                                       hdf5='./1ak4.hdf5',
	                                           )

	#create new files
	if not os.path.isfile(database.hdf5):
		t0 = time()
		print('{:25s}'.format('Create new database') + database.hdf5)
		database.create_database()
		print(' '*25 + '--> Done in %f s.' %(time()-t0))
	else:
		print('{:25s}'.format('Use existing database') + database.hdf5)

	# map the features
	grid_info = {
		'number_of_points' : [30,30,30],
		'resolution' : [1.,1.,1.],
		'atomic_densities' : {'CA':3.5,'N':3.5,'O':3.5,'C':3.5},
		'atomic_densities_mode' : 'sum',
		'atomic_feature_mode': 'sum'
	}
	t0 =time()
	print('{:25s}'.format('Map features in database') + database.hdf5)
	database.map_features(grid_info,try_sparse=True,time=False,prog_bar=False)
	print(' '*25 + '--> Done in %f s.' %(time()-t0))

if __name__ == "__main__":
	test_generate()


