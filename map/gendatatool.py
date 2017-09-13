import numpy as np 
import os
import subprocess as sp 
import sys

from deeprank.map import gridtool as gt

'''
	Generate the input/output data on the grids for a series of prot-prot conformations
	The calculation is actually performed by the gridtools class in GridTools.py

	ARGUMENTS:

	data_folder 

			main folder containing subfolder with pdbs targets/features
			of the complexes required for the dataset

	grid info

			dictionay containing the grid information
			see gridtool.py for details

	reset
			Boolean to force the removal of all data
'''

def map_features(data_folder, grid_info,data_type='haddock',reset=False):


	# check all the input PDB files
	sub_names = sp.check_output("ls -d %s/*/" %(data_folder),shell=True)
	sub_names = sub_names.split()

	# copy the cuve generating file in the folder
	#sp.call('cp generate_cube_file.py %s' %data_folder)


	# loop over the data files
	for isub,sub_ in enumerate(sub_names):

		# molecule name we want
		sub = sub_.decode('utf-8')

		# remove the data if we wnat to force that
		if os.path.isdir(sub+'/input') and reset:
			os.system('rm -rf %s' %(sub+'/input'))

		if not os.path.isdir(sub+'/input'):
			os.mkdir(sub+'/input/')
		

		# molecule name
		mol_name = sub + './complex.pdb'

		# create the residue feature dictionnary
		if 'residue_feature' in grid_info:
			res_feat = {}
			for feat_name in grid_info['residue_feature']:
				feat_file = sp.check_output("ls %s/%s/*" %(sub,feat_name),shell=True)
				res_feat[feat_name] = [f.decode('utf-8') for f in feat_file.split()]
		else:
			res_feat = None

		# compute the data we want on the grid
		grid = gt.GridTools(mol_name=mol_name,
						 data_type = data_type,
			             number_of_points = grid_info['number_of_points'],
			             resolution = grid_info['resolution'],
			             atomic_densities=grid_info['atomic_densities'],
			             residue_feature = res_feat,
			             export_path = sub+'/input/')



if __name__ == "__main__":

	# example use
	grid_info = {
		'atomic_densities' : {'CA':3.5,'CB':3.5,'N':3.5},
		'number_of_points' : [30,30,30],
		#'residue_feature' : ['PSSM'],
		'resolution' : [1.,1.,1.]
	}

	# main folder
	folder ='/home/nico/Documents/projects/deeprank/deeprank_classifier/test_set/'

	# go
	map_features(folder,grid_info)
