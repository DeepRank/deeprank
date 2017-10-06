import deeprank.map

##########################################################################
#
# STEP 2 MAP THE FEATURES
#
##########################################################################

# adress of the database
database = '../database_test/'


#define the dictionary for the grid
#many more options are available
#see deeprank/map/gridtool_sql.py

grid_info = {
	'atomic_densities' : {'CE':3.5,'CB':3.5,'CD':3.5},
	'atomic_densities_mode' : 'diff',
	'number_of_points' : [30,30,30],
	#'residue_feature' : ['PSSM'],
#	'atomic_feature'  : ['ELEC','VDW'],
	#'atomic_feature'  : ['CHARGE'],
	#'atomic_feature_mode': 'ind',
	'resolution' : [1.,1.,1.]
}


#map the features
deeprank.map.map_features(database,grid_info,use_tmpdir=True)

#visualize the data of one complex
#deeprank.#map.generate_viz_files(database+'/1AK4')


