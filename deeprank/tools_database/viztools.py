#!/usr/bin/env python

import numpy as  np
import argparse
import subprocess as sp 
import os
import pickle 
import h5py 
from deeprank.tools import pdb2sql 
from deeprank.tools import sparse

def create3Ddata(mol_name,molgrp):

	outdir = './_tmp_' + mol_name + '/'
		
	if not os.path.isdir(outdir):
		os.mkdir(outdir)

	# create the pdb file
	pdb_name = outdir + 'complex.pdb'
	if not os.path.isfile(pdb_name):
		sqldb = pdb2sql(molgrp['complex'].value)
		sqldb.exportpdb(pdb_name)
		sqldb.close()

	# get the grid
	grid = {}
	grid['x'] = molgrp['grid_points/x'].value
	grid['y'] = molgrp['grid_points/y'].value
	grid['z'] = molgrp['grid_points/z'].value
	shape = (len(grid['x']),len(grid['y']),len(grid['z']))

	# deals with the features
	mapgrp = molgrp['mapped_features']

	# loop through all the features
	for data_name in mapgrp.keys():

		# create a dict of the feature {name : value}
		featgrp = mapgrp[data_name]
		data_dict = {}
		for ff in featgrp.keys():
			subgrp = featgrp[ff]
			if not subgrp.attrs['sparse']:
				data_dict[ff] =  subgrp['value'].value 
			else:
				spg = sparse.FLANgrid(sparse=True,index=subgrp['index'].value,value=subgrp['value'].value,shape=shape)
				data_dict[ff] =  spg.to_dense()
				
		# export the cube file
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
		if not os.path.isfile(fname):
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

def launchVMD(mol_name,res):

	export_path =  './_tmp_' + mol_name + '/'
	exec_fname = 'loadData.vmd'

	# export VMD script if cube format is required		
	fname = export_path + exec_fname
	f = open(fname,'w')
	f.write('# can be executed with vmd -e loadData.vmd\n\n')
	
	# write all the cube file in one given molecule
	cube_files = list(filter(lambda x: '.cube' in x,os.listdir(export_path)))

	write_molspec_vmd(f, cube_files[0],'VolumeSlice','Volume')
	for idata in range(1,len(cube_files)):
		f.write('mol addfile ' + '%s\n' %(cube_files[idata]))
	f.write('mol rename top grid_data')

	# load the complex
	write_molspec_vmd(f,'complex.pdb','Cartoon','Chain')		

	# close file
	f.close()

	# launch VMD
	w,h = res.width()/2,res.height()/2
	sw,sh = 1050,600
	w,h = 600,600
	vmd_option = '-pos %f %f -size %f %f' %(sw,sh,w,h)
	vmd_file = ' -e ' + exec_fname
	sp.Popen('vmd ' + vmd_option + vmd_file, cwd = export_path,shell = True)
	
# quick shortcut for writting the vmd file
def write_molspec_vmd(f,name,rep,color):
	f.write('\nmol new %s\n' %name)
	f.write('mol delrep 0 top\nmol representation %s\n' %rep)
	if color is not None:
		f.write('mol color %s \n' %color)
	f.write('mol addrep top\n\n')



def launchPyMol(mol_name):


	export_path =  './_tmp_' + mol_name + '/'
	exec_fname = 'loadData.py'

	fname = export_path + exec_fname
	f = open(fname,'w')
	f.write('# can be executed with pymol -qRr loadData.py\n\n')
	f.write('import os\n')
	f.write('import pymol\n')
	f.write('pymol.finish_launching()\n\n')

	f.write("# load the molecule\n")
	f.write("pymol.cmd.load('complex.pdb','complex')\n")
	f.write("pymol.util.cbc(selection='(all)',first_color=7,quiet=1,legacy=0,_self=pymol.cmd)\n")
	f.write("pymol.cmd.show('stick','complex')\n\n")

	f.write("# load the molecule\n")
	f.write("cube_files = list(filter(lambda x: '.cube' in x,os.listdir('./')))\n")

	# load the cube files
	f.write("for f in cube_files:\n")
	f.write("	fname = os.path.splitext(f)[0]\n")
	f.write("	pymol.cmd.load(f)\n")
	f.write("	pymol.cmd.isosurface('pos_'+fname,fname,level=0.05)\n")
	f.write("	pymol.cmd.isosurface('neg_'+fname,fname,level=-0.05)\n\n")

	f.write("pymol.cmd.disable('all')\n")
	f.write("pymol.cmd.enable('complex')\n\n")

	f.close()

	sp.Popen('pymol -qQr ' + exec_fname, cwd = export_path,shell = True)
