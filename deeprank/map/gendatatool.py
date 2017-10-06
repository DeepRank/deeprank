import numpy as np 
import os
import subprocess as sp 
import sys

from deeprank.map import gridtool_sql as gt

def map_features(data_folder, grid_info,reset=False,use_tmpdir=False):

	'''
	Generate the input/output data on the grids for a series of prot-prot conformations
	The calculation is actually performed by the gridtools class in GridTools.py

	ARGUMENTS:

	data_folder 

			main folder containing subfolder with pdbs targets/features
			of the complexes required for the dataset

	grid info

			dictionay containing the grid information
			see gridtool_sql.py for details

	reset
			Boolean to force the removal of all data


	use_tmpdir
			Use the tmp dir to export the data 
			to avoid transferring files betwene computing and head nodes

	'''

	# check all the input PDB files
	sub_names = sp.check_output("ls -d %s/*/" %(data_folder),shell=True)
	sub_names = sub_names.split()

	# determine the atomic densities parametres
	if 'atomic_densities' in grid_info:
		atomic_densities = grid_info['atomic_densities']
	else:
		atomic_densities = None

	if 'atomic_densities_mode' in grid_info:
		atomic_densities_mode = grid_info['atomic_densities_mode']
	else:
		atomic_densities_mode = 'sum'


	# determine where to export
	if use_tmpdir:
		data_base = os.environ['TMPDIR']
		os.mkdir(data_base)

	# loop over the data files
	for isub,sub_ in enumerate(sub_names):

		# molecule name we want
		sub = sub_.decode('utf-8')

		# determine where to export
		if use_tmpdir:
			export_dir = data_base + sub.split('/')[-1]
			os.mkdir(export_dir)
			os.mkdir(export_dir+'/input/')

		else:

			# remove the data if we wnat to force that
			if os.path.isdir(sub+'/input') and reset:
				os.system('rm -rf %s' %(sub+'/input'))

			# create the input subfolder
			if not os.path.isdir(sub+'/input'):
				os.mkdir(sub+'/input/')
		
			# set the export dir
			export_dir = sub

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

		# create the atomic feature dictionary
		if 'atomic_feature' in grid_info:
			at_feat = {}
			for feat_name in grid_info['atomic_feature']:
				feat_file = sp.check_output("ls %s/%s/*" %(sub,feat_name),shell=True).decode('utf-8').split()
				if len(feat_file)>1:
					print('Warning: Multiple files found in %s.\nOnly considering the first one' %(sub))
				at_feat[feat_name] = feat_file[0]
		else:
			at_feat = None


		# compute the data we want on the grid
		grid = gt.GridToolsSQL(mol_name=mol_name,
			             number_of_points = grid_info['number_of_points'],
			             resolution = grid_info['resolution'],
			             atomic_densities = atomic_densities,
			             atomic_densities_mode = atomic_densities_mode,
			             residue_feature = res_feat,
			             atomic_feature = at_feat,
			             export_path = export_dir+'/input/')


