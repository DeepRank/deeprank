import os 
import sys
import importlib
import numpy as np
import subprocess as sp
import h5py

from deeprank.tools import pdb2sql
from deeprank.generate import GridTools as gt

try:
	from tqdm import tqdm
except:
	def tqdm(x):
		return x

'''
	Assemble the data set from different sources of  decoys/natives/features/targets

	ARGUMENTS

	pdb_select

			file containing the name of specfic complexe we want 
			in the databas

	pdb_source

			path or list of path where to find the pdbs


	data_augmentation

			None or integers
			if integers (N), each compound will be copied N times
			each copy having a different rotation randomly defined

	outdir

			directory where to output the database

'''

class DataGenerator(object):

	def __init__(self,pdb_select=None,pdb_source=None,pdb_native=None,
				 compute_targets = None, import_targets = None,
				 compute_features = None,import_features = None,
				 grid_info = None,
		         data_augmentation=None, hdf5='database.h5' ):

		self.pdb_select  = pdb_select
		self.pdb_source  = pdb_source
		self.pdb_native  = pdb_native

		self.data_augmentation = data_augmentation
		
		self.hdf5 = hdf5 

		self.compute_targets  = compute_targets
		self.import_targets = import_targets

		self.compute_features = compute_features
		self.import_features =  import_features

		self.grid_info = grid_info

		self.all_pdb = []
		self.all_native = []
		self.pdb_path = []

		# check that a source was given
		if self.pdb_source is None:
			raise NotADirectoryError('You must provide one or several source directory where the pdbs are stored')
			
		# handle the sources
		if not isinstance(self.pdb_source,list):
			self.pdb_source = [self.pdb_source]

		# get all the conformation path
		for src_dir in self.pdb_source:
			self.all_pdb += sp.check_output('find %s -name "*.pdb"' %src_dir,shell=True).decode('utf8').split()
			

		# handle the native
		if not isinstance(self.pdb_native,list) and self.native is not None:
			self.native = [self.native]

		for src_dir in self.pdb_native:
			self.all_native +=  sp.check_output('find %s -name "*.pdb"' %src_dir,shell=True).decode('utf8').split()

		# filter the cplx if required 
		self.pdb_path = self.all_pdb
		if self.pdb_select is not None:
			self._filter_cplx()

#====================================================================================
#
#		CREATE THE DATABASE ALL AT ONCE IF ALL OPTIONS ARE GIVEN
#
#====================================================================================

	def create_database(self,verbose=False):

		print(': Create hdf5 file')

		# open the file
		self.f5 = h5py.File(self.hdf5,'w')

		# loop over the decoys/natives
		for cplx in tqdm(self.pdb_path):


			################################################
			#	get the pdbs of the conformation and its ref
			#   for the original data (not augmetned one)
			################################################

			# names of the molecule
			mol_name = os.path.splitext(os.path.basename(cplx))[0]

			# get the bare name of the molecule
			# and define the name of the native
			# i.e. 1AK4_100w -> 1AK4
			bare_mol_name = mol_name.split('_')[0]
			ref_name = bare_mol_name + '.pdb'

			# check if we have a decoy or native
			# and find the reference
			if mol_name == bare_mol_name:
				ref = cplx
			else:
				ref = list(filter(lambda x: ref_name in x,self.all_native))
				if len(ref)>1:
					raise Warning('Multiple native complexes found for',mol_name)
				ref = ref[0]
				if ref == '':
					ref = None
				
			# talk a bit
			if verbose:
				print('\n: Process complex %s' %(mol_name))

			# crete a subgroup for the molecule
			molgrp = self.f5.require_group(mol_name)

			# add the ref and the complex
			self._add_pdb(molgrp,cplx,'complex')
			if ref is not None:
				self._add_pdb(molgrp,ref,'native')

			################################################
			#	add the features
			################################################

			# add the features
			featgrp = molgrp.require_group('features')
			if self.import_features is not None:
				self._import_features(self.import_features,featgrp)			

			if self.compute_features is not None:
				self._compute_features(self.compute_features, molgrp['complex'].value,molgrp['features'] )

			################################################
			#	add the targets
			################################################

			# add the features
			molgrp.require_group('targets')
			if self.import_targets is not None:
				self._import_targets(self.import_targets,mol_name)			

			if self.compute_targets is not None:
				self._compute_targets(self.compute_targets, molgrp['complex'].value,molgrp['targets'])	

			################################################
			#	DATA AUGMENTATION
			################################################			

			# GET ALL THE NAMES
			if self.data_augmentation is not None:
				mol_aug_name_list = [mol_name + '_r%03d' %(idir+1) for idir in range(self.data_augmentation)]
			else:
				mol_aug_name_list = []

			# loop over the complexes
			for icplx, mol_aug_name in enumerate(mol_aug_name_list):

				################################################
				#	get the pdbs of the conformation and its ref
				################################################

				# crete a subgroup for the molecule
				molgrp = self.f5.require_group(mol_aug_name)

				# copy the ref into it
				if ref is not None:
					self._add_pdb(molgrp,ref,'native')

				# create the new pdb
				self._add_aug_pdb(molgrp,cplx,'complex')

				# create the subgroups
				molgrp.require_group('targets')
				molgrp.require_group('features')

				# copy the targets/features
				molgrp.copy('targets',self.f5[mol_name+'/targets/'])
				molgrp.copy('features',self.f5[mol_name+'/features/'])					

		# close the file
		self.f5.close()

#====================================================================================
#
#		ADD FEATURES TO AN EXISTING DATASET
#
#====================================================================================

	def add_feature(self,compute_features=None,import_features=None):

		'''
		add a feature file to an existing folder arboresence
		only need an output dir and a feature dictionary
		'''
		print(': Add features')

	
		# get the folder names
		f5 = h5py.File(self.hdf5,'a')
		fnames = f5.keys()

		# get the non rotated ones
		fnames_original = list( filter(lambda x: '_r' not in x, fnames) )
		fnames_augmented = list( filter(lambda x: '_r' in x, fnames) )

		# computes the features of the original
		for cplx_name in tqdm(fnames_original):

			# molgrp
			molgrp = f5[cplx_name]

			# external features that are read from files
			if import_features is not None:
				self._import_features(import_features,molgrp)

			# the internal features
			if compute_features is not None:
				featgrp = molgrp['features']
				self._compute_features(compute_features,featgrp)

		# copy the data from the original to the augmented
		for cplx_name in fnames_augmented:

			# group of the molecule
			molgrp = f5[cplx_name]

			# copy the targets to the augmented
			mol_name = molgrp.name.split('_r')[0]
			molgrp.copy('features',self.f5[mol_name+'/features/'])


#====================================================================================
#
#		ADD TARGETS TO AN EXISTING DATASET
#
#====================================================================================

	def add_target(self,compute_targets=None,import_targets=None):

		'''
		add a target files to an existing folder arboresence
		only need an output dir and a target dictionary
		'''

		print(': Add targets')


		# name of the hdf5 file
		f5 = h5py.File(self.hdf5,'a')
						
		# get the folder names
		fnames = f5.keys()

		# get the non rotated ones
		fnames_original = list( filter(lambda x: '_r' not in x, fnames) )
		fnames_augmented = list( filter(lambda x: '_r' in x, fnames) )

		# compute the targets  of the original
		for cplx_name in tqdm(fnames_original):

			# group of the molecule
			molgrp = f5[cplx_name]

			# external_targets
			if import_targets is not None:
				self._import_targets(import_targets, molgrp )

			if compute_targets is not None:
				targrp = molgrp['targets']
				self._compute_targets(compute_targets,targrp)

		# copy the targets of the original to the rotated
		for cplx_name in fnames_augmented:

			# group of the molecule
			molgrp = f5[cplx_name]

			# copy the targets to the augmented
			mol_name = molgrp.name.split('_r')[0]
			molgrp.copy('targets',self.f5[mol_name+'/targets/'])

		# close the file
		file_hdf5.close()

#====================================================================================
#
#		MAP THE FEATURES TO THE GRID
#
#====================================================================================


	def map_features(self,grid_info,reset=False,use_tmpdir=False):

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
				to avoid transferring files between computing and head nodes

		'''


		# name of the hdf5 file
		f5 = h5py.File(self.hdf5,'a')

		# check all the input PDB files
		mol_names = f5.keys()
		
		# fills in the grid data if not provided : default = NONE
		grinfo = ['number_of_points','resolution']
		for gr in grinfo:
			if gr not in grid_info:
				grid_info[gr] = None

		# fills in the features if somes are missing : default = NONE
		inp_data = ['atomic_densities','atomic_feature','residue_feature']
		for inp in inp_data:
			if inp not in grid_info:
				grid_info[inp] = None

		# fills in the features mode if somes are missing : default = SUM
		modes = ['atomic_densities_mode','atomic_feature_mode','residue_feature_mode']
		for m in modes:
			if m not in grid_info:
				grid_info[m] = 'sum'


		# loop over the data files
		for mol in mol_names:
					
			# compute the data we want on the grid
			grid = gt.GridTools(molgrp=f5[mol],
				             number_of_points = grid_info['number_of_points'],
				             resolution = grid_info['resolution'],
				             atomic_densities = grid_info['atomic_densities'],
				             atomic_densities_mode = grid_info['atomic_densities_mode'],
				             residue_feature = grid_info['residue_feature'],
				             atomic_feature = grid_info['atomic_feature'],
				             atomic_feature_mode = grid_info['atomic_feature_mode'],
				             hdf5_file = f5)

		# close he hdf5 file
		f5.close()

#====================================================================================
#
#		FILTER DATASET
#
#===================================================================================

	def _filter_cplx(self):

		# read the class ID
		f = open(self.pdb_select)
		pdb_name = f.readlines()
		f.close()
		pdb_name = [name.split()[0]+'.pdb' for name in pdb_name]

		# create the filters
		tmp_path = []
		for name in pdb_name:	
			tmp_path += list(filter(lambda x: name in x,self.pdb_path))

		# update the pdb_path
		self.pdb_path = tmp_path
		


#====================================================================================
#
#		FEATURES ROUTINES
#
#====================================================================================

	def _import_features(self,source_files,molgrp):

		# get all the features
		for src in source_files:

			fsrc  = h5py.File(src,'r')
			if molgrp.name in fsrc.keys:
				molgrp.require_group('features')
				molgrp.copy('features',fsrc[molgrp.name+'/features/']) 
				fsrc.close()
				break
			else:
				fsrc.close()

	def _compute_features(self,feat_list,pdb_data,featgrp):

		for feat in feat_list:
			feat_module = importlib.import_module(feat,package=None)
			feat_module.__compute_feature__(pdb_data,featgrp)


#====================================================================================
#
#		TARGETS ROUTINES
#
#====================================================================================

	def _import_targets(self,source_files,molgrp):

		# get all the features
		for src in source_files:

			fsrc  = h5py.File(src,'r')
			if molgrp.name in fsrc.keys:
				molgrp.require_group('targets')
				molgrp.copy('targets',fsrc[molgrp.name+'/targets/']) 
				fsrc.close()
				break
			else:
				fsrc.close()

	def _compute_targets(self,targ_list,pdb_data,targrp):

		for targ in targ_list:
			targ_module = importlib.import_module(targ,package=None)
			targ_module.__compute_target__(pdb_data,targrp)


#====================================================================================
#
#		ADD PDB FILE
#
#====================================================================================

	def _add_pdb(self,molgrp,pdbfile,name):

		# read the pdb and extract the ATOM lines
		with open(pdbfile,'r') as fi:
			data = [line.split('\n')[0] for line in fi if line.startswith('ATOM')]
		data = np.array(data).astype('|S73')
		dataset = molgrp.create_dataset(name,data=data)


	def _add_aug_pdb(self,molgrp,pdbfile,name):


		# create tthe sqldb and extract positions
		sqldb = pdb2sql(pdbfile)
		xyz = sqldb.get('x,y,z')

		# define the transformation axis
		axis = -1 + 2*np.random.rand(3)
		axis /= np.linalg.norm(axis)

		# define the axis
		# uniform distribution on a sphere
		# http://mathworld.wolfram.com/SpherePointPicking.html
		u1,u2 = np.random.rand(),np.random.rand()
		teta,phi = np.arccos(2*u1-1),2*np.pi*u2
		axis = [np.sin(teta)*np.cos(phi),np.sin(teta)*np.sin(phi),np.cos(teta)]

		# and the rotation angle
		angle = -np.pi + np.pi*np.random.rand()

		# rotate the positions
		sqldb.rotation_around_axis(axis,angle)
		
		# get the data
		sqldata = sqldb.get('*')

		# close the db
		sqldb.close()

		# export the data to h5
		data = []
		for d in sqldata:
			line = 'ATOM  '
			line += '{:>5}'.format(d[0])	# serial
			line += ' '
			line += '{:^4}'.format(d[1])	# name
			line += '{:>1}'.format(d[2])	# altLoc
			line += '{:>3}'.format(d[3])	#resname
			line += ' '
			line += '{:>1}'.format(d[4])	# chainID
			line += '{:>4}'.format(d[5])	# resSeq
			line += '{:>1}'.format(d[6])	# iCODE
			line += '   '
			line += '{: 8.3f}'.format(d[7])	#x
			line += '{: 8.3f}'.format(d[8])	#y
			line += '{: 8.3f}'.format(d[9])	#z
			line += '{: 6.2f}'.format(d[10])	# occ
			line += '{: 6.2f}'.format(d[11])	# temp
			data.append(line)

		data = np.array(data).astype('|S73')
		dataset = molgrp.create_dataset(name,data=data)