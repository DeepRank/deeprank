import deeprank

##########################################################################
#
# STEP 2 MAP THE FEATURES
#
##########################################################################

# adress of the database
database = './training_set/'


#define the dictionary for the grid
#many more options are available
#see deeprank/map/gridtool.py

grid_info = {
	'atomic_densities' : {'CA':3.5,'CB':3.5,'N':3.5},
	'number_of_points' : [30,30,30],
	'resolution' : [1.,1.,1.]
}


#map the features
deeprank.map_features(database,grid_info)

#visualize the data of one complex
deeprank.generate_viz_files(database+'/1AK4')


