import deeprank.map

##########################################################################
#
# STEP 2 MAP THE FEATURES
#
##########################################################################

# adress of the database
database = '../../database/'


#define the dictionary for the grid
#many more options are available
#see deeprank/map/gridtool_sql.py

grid_info = {
	'atomic_densities' : {'CD':3.5,'CE':3.5,'O':3.5},
	'atomic_densities_mode' : 'diff',
	'number_of_points' : [30,30,30],
	#'residue_feature' : ['PSSM'],
	'atomic_feature'  : ['ELEC','VDW','CHARGE'],
	'resolution' : [1.,1.,1.]
}


#map the features
deeprank.map.map_features(database,grid_info)

#visualize the data of one complex
deeprank.map.generate_viz_files(database+'/1AK4')


