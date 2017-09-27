import deeprank.map
import os 

##########################################################################
#
# Use the GridTool class to map the features of one complex only
# and create the cube files and VMD scripts to visulize the data
#
##########################################################################

if os.path.isdir('./gridtest/input'):
	os.system('rm -rf ./gridtest/input')

if os.path.isdir('./gridtest/data_viz'):
	os.system('rm -rf ./gridtest/data_viz')


grid = deeprank.map.GridToolsSQL(mol_name='./gridtest/complex.pdb',
	             number_of_points = [30,30,30],
	             resolution = [1.,1.,1.],
	             atomic_densities={'CA':3.5, 'CB':3.5},
	             residue_feature={
	             'PSSM' : './gridtest/PSSM/1AK4.PSSM'},
	             export_path = './gridtest/input/')

#visualize the data of one complex
deeprank.map.generate_viz_files('./gridtest/')