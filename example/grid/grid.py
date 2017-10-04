import deeprank.map
import os 

##########################################################################
#
# Use the GridTool class to map the features of one complex only
# and create the cube files and VMD scripts to visulize the data
#
##########################################################################

if os.path.isdir('./input'):
	os.system('rm -rf ./input')

if os.path.isdir('./data_viz'):
	os.system('rm -rf ./data_viz')


grid = deeprank.map.GridToolsSQL(mol_name='./complex.pdb',
	             number_of_points = [30,30,30],
	             resolution = [1.,1.,1.],
	             atomic_densities={'CA' : 3.5, 'N' : 3.5, 'CB':3.5},
	             atomic_densities_mode = 'diff',
	             #residue_feature={
	             #'PSSM' : './PSSM/1AK4.PSSM'},
	             atomic_feature={
	             'coulomb' : './ELEC/complex.COULOMB',
	             'vdw' : './VDW/complex.VDWAALS'
	             },
	             export_path = './input/')

#visualize the data of one complex
deeprank.map.generate_viz_files('./')