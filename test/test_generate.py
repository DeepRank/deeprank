import sys
import deeprank.generate
import os
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
		database.create_database()
	else:
		print('\nUsing peviously generated database : %s' database.hdf5)

	# map the features
	grid_info = {
		'atomic_densities' : {'CA':3.5,'N':3.5,'O':3.5,'C':3.5},
		'atomic_densities_mode' : 'sum',
		'number_of_points' : [30,30,30],
		'atomic_feature'  : ['vdwaals','coulomb','charge'],
		'atomic_feature_mode': 'sum',
		'resolution' : [1.,1.,1.]
	}
	database.map_features(grid_info,time=True,prog_bar=False)


if __name__ == "__main__":
	test_generate()


