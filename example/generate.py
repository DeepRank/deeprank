from deeprank.generate import *


##########################################################################
#
#	GENERATE THE DATA BASE AT ONCE
#	--> assemble the pdbs
#   --> compute the features on the fly 
#   --> compute the targets on the fly 
#   --> map the features on the grid
#
##########################################################################

# adress of the BM4 folder
BM4 = '/home/nico/Documents/projects/deeprank/data/HADDOCK/BM4_dimers/'


# sources to assemble the data base
pdb_source     = [BM4 + 'decoys_pdbFLs/1AK4/water/']
pdb_native     = [BM4 + 'BM4_dimers_bound/pdbFLs_ori']

#init the data assembler 
database = DataGenerator(pdb_source=pdb_source,pdb_native=pdb_native,data_augmentation=None,
                        compute_targets  = ['deeprank.tools.targets.dockQ'],
                        compute_features = ['deeprank.tools.features.atomic,deeprank.tools.features.pssm'],
                        hdf5='./1ak4.hdf5',
                        )
 
#create new files
database.create_database()

# map the features
grid_info = {
	'number_of_points' : [30,30,30],
	'resolution' : [1.,1.,1.],
	'atomic_densities' : {'CA':3.5,'N':3.5,'O':3.5,'C':3.5},
	'atomic_densities_mode' : 'diff',
	'feature_mode': 'sum'
}

database.map_features(grid_info)




