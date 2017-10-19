import sys
import deeprank.generate


##########################################################################
#
#	GENERATE THE DATA BASE AT ONCE
#	--> assemble the pdbs
#   --> compute the features on the fly or import them 
#   --> compute the targets on the fly or import them 
#   --> map the features on the grid
#
##########################################################################

# adress of the BM4 folder
BM4 = '/home/nico/Documents/projects/deeprank/data/HADDOCK/BM4_dimers/'
#BM4 = sys.argv[1]

# sources to assemble the data base
pdb_source     = [BM4 + 'decoys_pdbFLs/', BM4 + '/BM4_dimers_bound/pdbFLs_refined']
pdb_select     =  BM4 + '/training_set_IDS/trainIDs.lst'

# adress of the database
outdir = '../../database_refined_test/'

#inti the data assembler 
database = deeprank.generate.DataGenerator(pdb_select=pdb_select,
	                                       pdb_source=pdb_source,
	                                       outdir=outdir,
                                           data_augmentation=None)

#create new files
database.create_database()


# add internal/external features 
internal_features = ['deeprank.tools.atomic_features']
external_features = {'ELEC' : BM4+'/ELEC'}
database.add_feature(internal=internal_features,external=external_features)


#add new targets
internal_targets = ['deeprank.tools.StructureSimilarity','deeprank.tools.binary_class']
external_targets = {'haddock_score' : BM4 + '/model_qualities/haddockScore/water'}
database.add_target(internal=internal_targets,external=external_targets)


# map the features
grid_info = {
	'atomic_densities' : {'CA':3.5,'N':3.5,'O':3.5},
	'atomic_densities_mode' : 'diff',
	'number_of_points' : [30,30,30],
	'atomic_feature'  : ['VDWAALS','COULOMB','CHARGE'],
	'atomic_feature_mode': 'sum',
	'resolution' : [1.,1.,1.]
}

database.map_features(grid_info)




