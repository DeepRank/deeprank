#!/usr/bin/env python

import deeprank.generate
import h5py
import os

def clean_dataset(fname,feature=True,pdb=True,points=True,grid=False):

	# name of the hdf5 file
	f5 = h5py.File(fname,'a')

	# get the folder names
	mol_names = f5.keys()

	for name in mol_names:

		mol_grp = f5[name]

		if feature and 'features' in mol_grp:
			del mol_grp['features']
		if pdb and 'complex' in mol_grp and 'native' in mol_grp:
			del mol_grp['complex']
			del mol_grp['native']
		if points and 'grid_points' in mol_grp:
			del mol_grp['grid_points']
		if grid and 'mapped_features' in mol_grp:
			del mol_grp['mapped_features']

	f5.close()

	os.system('h5repack %s _tmp.h5py' %fname)
	os.system('mv _tmp.h5py %s' %fname)

if __name__ == '__main__':

	import argparse
	import os

	parser = argparse.ArgumentParser(description='remove data from a hdf5 data set')
	parser.add_argument('hdf5', help="hdf5 file storing the data set",default=None)
	parser.add_argument('--keep_feature', action='store_true',help="keep the features")
	parser.add_argument('--keep_pdb', action='store_true',help="keep the pdbs")
	parser.add_argument('--keep_pts',action='store_true',help="keep the coordinates of the grid points")
	parser.add_argument('--rm_grid',action='store_true',help='remove the mapped feaures on the grids')
	args = parser.parse_args()

	clean_dataset(args.hdf5,
		         feature = not args.keep_feature,
		         pdb = not args.keep_pdb,
		         points = not args.keep_pts,
		         grid = args.rm_grid )

