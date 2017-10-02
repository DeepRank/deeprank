#!/usr/bin/env python

import numpy as  np
import argparse
import subprocess as sp 
import os
import pickle 

def generate_viz_files(mol_dir):


	'''
	This function can be used to generate cube files for the visualization of the mapped 
	data in VMD

	Usage
	python generate_cube_files.py <mol_dir_name>
	e.g. python generate_cube_files.py 1AK4

	or within a python script

	import deeprank.map
	deeprank.map.generate_viz_files(mol_dir_name)

	A new subfolder data_viz will be created in <mol_dir_name>
	with all the cube files representing the features contained in 
	the files <mol_dir_name>/input/*.npy

	A script called <feature_name>.vmd is also outputed et allow for 
	quick vizualisation of the data by typing

	vmd -e <feature_name>.vmd
	'''

	# create the output directory
	outdir = mol_dir+'/data_viz/'
	if not os.path.isdir(outdir):
		os.mkdir(outdir)

	# make a copy of the pdb file
	if os.path.isfile(mol_dir+'/complex.pdb'):
		sp.call('cp %s/complex.pdb %s' %(mol_dir,outdir),shell=True)

	# get the grid points
	grid = np.load(mol_dir+'/input/grid_points.npz')

	# get all the data file in the directory
	#fnames_npy = sp.check_output('ls %s/input/*.npy' %(mol_dir),shell=True).decode('utf8').split()
	fnames_pkl = sp.check_output('ls %s/input/*.pkl' %(mol_dir),shell=True).decode('utf8').split()

	# loop over all the data files
	for f in fnames_pkl:

		data_dict = pickle.load(open(f,'rb'))
		data_name = f.split('/')[-1][:-4]
		export_cube_files(data_dict,data_name,grid,outdir)

def export_cube_files(data_dict,data_name,grid,export_path):

	print('-- Export %s data to %s' %(data_name,export_path))
	bohr2ang = 0.52918

	# individual axis of the grid
	x,y,z = grid['x'],grid['y'],grid['z']

	# extract grid_info
	npts = np.array([len(x),len(y),len(z)])
	res = np.array([x[1]-x[0],y[1]-y[0],z[1]-z[0]])

	# the cuve file is apparently give in bohr
	xmin,ymin,zmin = np.min(x)/bohr2ang,np.min(y)/bohr2ang,np.min(z)/bohr2ang
	scale_res = res/bohr2ang

	# export files for visualization 
	for key,values in data_dict.items():

		fname = export_path + data_name + '_%s' %(key) + '.cube'
		f = open(fname,'w')
		f.write('CUBE FILE\n')
		f.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
		
		f.write("%5i %11.6f %11.6f %11.6f\n" %  (1,xmin,ymin,zmin))
		f.write("%5i %11.6f %11.6f %11.6f\n" %  (npts[0],scale_res[0],0,0))
		f.write("%5i %11.6f %11.6f %11.6f\n" %  (npts[1],0,scale_res[1],0))
		f.write("%5i %11.6f %11.6f %11.6f\n" %  (npts[2],0,0,scale_res[2]))


		# the cube file require 1 atom
		f.write("%5i %11.6f %11.6f %11.6f %11.6f\n" %  (0,0,0,0,0))

		last_char_check = True
		for i in range(npts[0]):
			for j in range(npts[1]):
				for k in range(npts[2]):
					f.write(" %11.5e" % values[i,j,k])
					last_char_check = True
					if k % 6 == 5: 
						f.write("\n")
						last_char_check = False
				if last_char_check:
					f.write("\n")
		f.close() 


		# export VMD script if cube format is required		
		fname = export_path + data_name + '.vmd'
		f = open(fname,'w')
		f.write('# can be executed with vmd -e viz_mol.vmd\n\n')
		
		# write all the cube file in one given molecule
		keys = list(data_dict.keys())
		write_molspec_vmd(f, data_name +'_%s.cube' %(keys[0]),'VolumeSlice','Volume')
		for idata in range(1,len(keys)):
			f.write('mol addfile ' + data_name +'_%s.cube\n' %(keys[idata]))
		f.write('mol rename top ' + data_name)

		# load the complex
		write_molspec_vmd(f,'complex.pdb','Cartoon','SegName')		

		f.close()


# quick shortcut for writting the vmd file
def write_molspec_vmd(f,name,rep,color):
	f.write('\nmol new %s\n' %name)
	f.write('mol delrep 0 top\nmol representation %s\n' %rep)
	if color is not None:
		f.write('mol color %s \n' %color)
	f.write('mol addrep top\n\n')


if __name__ == "__main__":
	
	import argparse

	parser = argparse.ArgumentParser(description='export the grid data in cube format')
	parser.add_argument('mol_dir',help="Directory of the molecule")
	args = parser.parse_args()

	# shortcut
	mol_dir = args.mol_dir

	generate_viz_files(mol_dir)