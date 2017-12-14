#!/usr/bin/env python

import deeprank.generate
import h5py


def clean_dataset(fname,feature=True,pdb=True,grid=True):

	# name of the hdf5 file
	f5 = h5py.File(fname,'w')
					
	# get the folder names
	fnames = f5.keys()

	for name in fnames:
		if feature:
			del f5[name+'/features']
		if pdb:
			del f5[name+'/complex']
			del f5[name+'/native']
		if grid:
			del f5[name+'/grid_points']

	f5.close()

if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser(description='remove data from a hdf5 data set')
	parser.add_argument('hdf5', help="hdf5 file storing the data set",default=None)
	parser.add_argument('--keep_feature', action='store_true',help="keep the features")
	parser.add_argument('--keep_pdb', action='store_true',help="keep the pdbs")
	parser.add_argument('--keep_grid',action='store_true',help='keep the grid points')
	args = parser.parse_args()


	clean_dataset(args.hdf5,
		         feature = not args.keep_feature,
		         pdb = not args.keep_pdb,
		         grid = not args.keep_grid )



