import os 
import sys
import importlib
import numpy as np
import subprocess as sp

from deeprank.tools import pdb2sql
import deeprank.tools.transform as transf
from deeprank.map import gridtool_sql as gt

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

	def __init__(self,pdb_select=None,pdb_source=None,data_augmentation=None, outdir=None):

		self.pdb_select  = pdb_select
		self.pdb_source  = pdb_source
		self.data_augmentation = data_augmentation
		self.outdir = outdir

		self.all_pdb = []
		self.pdb_path = []


#====================================================================================
#
#		MAIN FUNCTIONS
#
#====================================================================================

	def create_database(self):

		'''
		Create all the data files generate by all the pdb contained
		in the natives + decoys directories
		'''

		# check if we can create a dir here
		self._check_outdir()

		# check that a source was given
		if self.pdb_source is None:
			raise NotADirectoryError('You must provide one or several source directory where the pdbs are stored')
			
		# handle the sources
		if not isinstance(self.pdb_source,list):
			self.pdb_source = [self.pdb_source]

		# get all the conformation path
		for src_dir in self.pdb_source:
			self.all_pdb += sp.check_output('find %s -name "*.pdb"' %src_dir,shell=True).decode('utf8').split()
			
		# filter the cplx if required 
		self.pdb_path = self.all_pdb
		if self.pdb_select is not None:
			self._filter_cplx()

		# create the data files
		self._create()


	def add_feature(self,internal=None,external=None):

		'''
		add a feature file to an existing folder arboresence
		only need an output dir and a feature dictionary
		'''
		print(': Add features')

		if not os.path.isdir(self.outdir):
			raise NotADirectoryError('%s is not a directory')

		# get the folder names
		fnames = sp.check_output('ls -d %s/*/' %self.outdir,shell=True).decode('utf8').split()


		# get the non rotated ones
		fnames_original = list( filter(lambda x: '_r' not in x, fnames) )
		fnames_augmented = list( filter(lambda x: '_r' in x, fnames) )

		# computes the features of the original
		for cplx_name in tqdm(fnames_original):

			# names of the molecule
			mol_name = cplx_name.split('/')[-2]

			# directory of the complex
			feat_dir_name = self.outdir + '/' + mol_name + '/features/'

			# external features that are read from files
			if external is not None:
				self._add_external_features(external,ref_mol_name,feat_dir_name)

			# the internal features
			if internal is not None:
				mol = cplx_name + '/complex.pdb'
				self._add_internal_features(internal,mol,feat_dir_name)

		# copy the data from the original to the augmented
		for cplx_name in fnames_augmented:

			# names of the molecule
			mol_name = cplx_name.split('/')[-2]
			ref_mol_name = mol_name.split('_r')[0]
			
			# input the data	
			target_dir_name = self.outdir + '/' + mol_name + '/features/'
			ref_dir_name = self.outdir + '/' + ref_mol_name + '/features/'	

			# copy the data
			sp.call('cp %s/* %s' %(ref_dir_name,target_dir_name),shell=True)	


	def add_target(self,internal=None,external=None):

		'''
		add a target files to an existing folder arboresence
		only need an output dir and a target dictionary
		'''

		print(': Add targets')

		if not os.path.isdir(self.outdir):
			raise NotADirectoryError('%s is not a directory',self.outdir) 
			
		# get the folder names
		fnames = sp.check_output('ls -d %s/*/' %self.outdir,shell=True).decode('utf8').split()

		# get the non rotated ones
		fnames_original = list( filter(lambda x: '_r' not in x, fnames) )
		fnames_augmented = list( filter(lambda x: '_r' in x, fnames) )

		# compute the features of the original
		for cplx_name in tqdm(fnames_original):

			# names of the molecule
			mol_name = cplx_name.split('/')[-2]

			# input the data	
			target_dir_name = self.outdir + '/' + mol_name + '/targets/'

			# external_targets
			if external is not None:
				self._add_external_targets(external,ref_mol_name,target_dir_name)

			if internal is not None:
				mol = cplx_name + '/complex.pdb'
				self._add_internal_targets(internal,mol,target_dir_name)

		# copy the targets of the original to the rotated
		for cplx_name in fnames_augmented:

			# names of the molecule
			mol_name = cplx_name.split('/')[-2]
			ref_mol_name = mol_name.split('_r')[0]
			
			# input the data	
			target_dir_name = self.outdir + '/' + mol_name + '/targets/'
			ref_dir_name = self.outdir + '/' + ref_mol_name + '/targets/'	

			# copy the data
			sp.call('cp %s/* %s' %(ref_dir_name,target_dir_name),shell=True)	

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
				to avoid transferring files betwene computing and head nodes

		'''

		# check all the input PDB files
		sub_names = sp.check_output("ls -d %s/*/" %(self.outdir),shell=True)
		sub_names = sub_names.split()

		# name of the dir where the pckl files are stores
		outname = '/grid_data'

		# determine the atomic densities parametres
		if 'atomic_densities' in grid_info:
			atomic_densities = grid_info['atomic_densities']
		else:
			atomic_densities = None

		if 'atomic_densities_mode' in grid_info:
			atomic_densities_mode = grid_info['atomic_densities_mode']
		else:
			atomic_densities_mode = 'sum'


		if 'atomic_feature_mode' in grid_info:
			atomic_feature_mode = grid_info['atomic_feature_mode']
		else:
			atomic_feature_mode = 'sum'

		# determine where to export
		if use_tmpdir:
			data_base = os.environ['TMPDIR']
			os.mkdir(data_base)

		# loop over the data files
		for isub,sub_ in enumerate(sub_names):

			# molecule name we want
			sub = sub_.decode('utf-8')
			sub_mol = list(filter(None,sub.split('/')))[-1]
			
			# determine where to export
			if use_tmpdir:
				export_dir = data_base + '/' + sub_mol
				os.mkdir(export_dir)
				os.mkdir(export_dir + outname )


			else:

				# remove the data if we want to force that
				if os.path.isdir(sub + outname) and reset:
					os.system('rm -rf %s' %(sub+outname))

				# create the input subfolder
				if not os.path.isdir(sub+outname):
					os.mkdir(sub+outname)
			
				# set the export dir
				export_dir = sub

			# molecule name
			mol_name = sub + './complex.pdb'

			# create the residue feature dictionnary
			if 'residue_feature' in grid_info:
				res_feat = {}
				for feat_name in grid_info['residue_feature']:
					feat_file = sp.check_output("ls %s/features/*.%s" %(sub,feat_name),shell=True)
					res_feat[feat_name] = [f.decode('utf-8') for f in feat_file.split()]
			else:
				res_feat = None

			# create the atomic feature dictionary
			if 'atomic_feature' in grid_info:
				at_feat = {}
				for feat_name in grid_info['atomic_feature']:
					feat_file = sp.check_output("ls %s/features/*.%s" %(sub,feat_name),shell=True).decode('utf-8').split()
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
				             atomic_feature_mode = atomic_feature_mode,
				             export_path = export_dir+outname)

#====================================================================================
#
#		CREATE THE DATABASE
#
#====================================================================================

	def _check_outdir(self):

		if os.path.isdir(self.outdir):
			print(': Database  %s already exists' %(self.outdir))
			print(': Adding new data to existing one')
		else:
			print(': New output directory created at %s' %(self.outdir))
			os.mkdir(self.outdir)



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
		

	def _create(self,verbose=False):

		print(': Create database')

		# loop over the decoys/natives
		for cplx in tqdm(self.pdb_path):

			# names of the molecule
			mol_name = cplx.split('/')[-1][:-4]
			bare_mol_name = mol_name.split('_')[0]
			ref_name = bare_mol_name + '.pdb'

			# check if we have a decoy or native
			# and find the reference
			if mol_name == bare_mol_name:
				ref = cplx
			else:
				ref = list(filter(lambda x: ref_name in x,self.all_pdb))
				if len(ref)>1:
					raise LookupError('Multiple native complexes found for',mol_name)
					ref = ref[0]
				if len(ref) == 0:
					ref = None

			# talk a bit
			if verbose:
				print('\n: Process complex %s' %(mol_name))

			# Assemble the list of subfolder names
			cplx_dir_name_list = [self.outdir + '/' + mol_name]

			# make the copies if required
			if self.data_augmentation is not None:
				cplx_dir_name_list += [self.outdir + '/' + mol_name + '_r%03d' %(idir+1) for idir in range(self.data_augmentation)]

			# loop over the complexes
			for icplx, cplx_dir_name in enumerate(cplx_dir_name_list):

				# create the dir
				os.mkdir(cplx_dir_name)

				# copy the pdb file in it
				new_cplx_file = '%s/complex.pdb' %cplx_dir_name
				sp.call('cp %s %s' %(cplx,new_cplx_file),shell=True)

				# copy the reference file into it 
				if ref is not None:
					new_ref_file = '%s/ref.pdb' %cplx_dir_name
					sp.call('cp %s %s' %(ref,new_ref_file),shell=True)

				# make the copy
				if icplx > 0:

					#print(':  --> rotation %03d/%03d' %(icplx,len(cplx_dir_name_list)-1))

					# create tthe sqldb and extract positions
					sqldb = pdb2sql(new_cplx_file)
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

					# print rotation data
					if verbose:
						print(':        axis : %s' %' '.join(map('{: 0.3f}'.format,axis)))
						print(':        angle : % 1.3f\n' %angle)

					# rotate the positions
					xyz = transf.rotation_around_axis(xyz,axis,angle)
					
					# input in the database
					sqldb.update_xyz(xyz)

					# export the new pdb
					sqldb.exportpdb(new_cplx_file)

					# close the db
					sqldb.close()

				# create the target and feature dirs 
				target_dir_name = cplx_dir_name + '/targets/'
				os.mkdir(target_dir_name)

				feature_dir_name = cplx_dir_name + '/features/'
				os.mkdir(feature_dir_name)


#====================================================================================
#
#		FEATURES FILES
#
#====================================================================================

	def _add_internal_features(self,feat_list,pdb,feat_dir_name):

		for feat in feat_list:
			feat_module = importlib.import_module(feat,package=None)
			feat_module.__compute_feature__(pdb,feat_dir_name)



	def _add_external_features(self,feat_dict,mol_name,feat_dir_name):

		# get all the features
		for feat_name,feat_dir in feat_dict.items():

			# the native part of the name only
			bare_mol_name = mol_name.split('_')[0]

			# get the names of the all the files in the source directory
			filenames = sp.check_output('ls %s' %(feat_dir),shell=True).decode('utf-8').split()
			filenames = [name.split('/')[-1].split('.')[0] for name in filenames]

			# copy all the files containing the bare mol name in that directory
			if mol_name in filenames:
				sp.call('cp %s/*%s.* %s/' %(feat_dir,mol_name,feat_dir_name),shell=True)
			elif bare_mol_name in filenames:
				sp.call('cp %s/*%s.* %s/' %(feat_dir,bare_mol_name,feat_dir_name),shell=True)
			else:
				print('Error: no %s file found %s' %(feat_name,mol_name))


#====================================================================================
#
#		TARGETS FILES
#
#====================================================================================

	def _add_external_targets(self,targ_dict,mol_name,target_dir_name,rename=False):

		for targ_name,targ_dir in targ_dict.items():

			if targ_name == 'copy':
				sp.call('cp %s/%s*  %s' %(targ_dir,mol_name,target_dir_name),shell=True)

			else:
				# find the correct value and print it
				try:
					tar_val = sp.check_output('grep -w %s %s/*.*' %(mol_name,targ_dir),shell=True).decode('utf8').split()[-1]
					np.savetxt(target_dir_name + targ_name + '.dat',np.array([float(tar_val)]))
				except:
					pass



	def _add_internal_targets(self,targ_list,pdb,target_dir_name):

		for targ in targ_list:
			targ_module = importlib.import_module(targ,package=None)
			targ_module.__compute_target__(pdb,target_dir_name)














