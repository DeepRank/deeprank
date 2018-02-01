import os
import sys
import time
import h5py
import pickle
from itertools import chain

import torch
from torch import FloatTensor
import torch.utils.data as data_utils

import subprocess as sp
import numpy as np

from deeprank.generate import NormParam,MinMaxParam,NormalizeData
from deeprank.tools import sparse

try:
	from tqdm import tqdm
except:
	def tqdm(x):
		return x

printif = lambda string,cond: print(string) if cond else None

class DataSet(data_utils.Dataset):

	'''
	Class that generates the data needed for deeprank.

	The data is stored in memory on the fly.
	That allows to handle large data set but might alters performance

	ARGUMENTS 
		
	database : string or list of strings

		Path of the HDF5 file(s) containing the database(s)
		This hdf5 file(s) must be generated with the deeprank.generate
		tools to insure the correct structure

	select_feature : dict or 'all'
		
		if 'all', all the mapped features contained in the HDF5 file
		will be loaded

		if a dict must be of the structure. E.g. :

		{name : [list of keys]} e.g. {AtomicDensities : ['CA','CB']}
		{name : 'all'}     e.g {feature : 'all'}

		the name must correspond to a valid group : f[mo/mapped_features/name]
		
		if the value is a list of keys only thoses keys will be loaded
		if 'all' all the data will be loaded

	select_target

		the name of the target we want. 
		the name must correspond to a valid group : f[mo/targets/name]

	USAGE

		data = DeepRankDataSet(hdf5_file)

	'''

	def __init__(self,database,test_database=None,
		         select_feature='all',select_target='DOCKQ',
		         pair_ind_feature = False,
		         transform_to_2D=False,projection=0,grid_shape = None,
		         normalize_features=True,normalize_targets=True,tqdm=False):


		# allow for multiple database
		self.database = database
		if not isinstance(database,list):
			self.database = [database]

		# allow for multiple database
		self.test_database = test_database
		if test_database is not None:
			if not isinstance(test_database,list):
				self.test_database = [test_database]

		# features/targets selection normalization
		self.select_feature = select_feature
		self.select_target = select_target

		# normalization conditions
		self.normalize_features = normalize_features
		self.normalize_targets = normalize_targets

		# final containers
		self.features = None
		self.targets  = None
		self.input_shape = None
		self.data_hape = None
		self.grid_shape = grid_shape

		# the possible pairing of the ind features
		self.pair_ind_feature = pair_ind_feature

		# get the eventual projection
		self.transform = transform_to_2D
		self.proj2D = projection

		# print the progress bar or not
		self.tqdm=tqdm


		print('\n')
		print('='*40)
		print('=\t DeepRank Data Set')
		print('=')
		print('=\t Training data' )
		for f in self.database:
			print('=\t ->',f)
		print('=')
		if self.test_database is not None:
			print('=\t Test data' )
			for f in self.test_database:
				print('=\t ->',f)
		print('=')
		print('='*40,'\n')		
		sys.stdout.flush()

		# create the indexing system
		# alows to associate each mol to an index
		# and get fname and mol name from the index
		self.create_index_molecules()

		# get the actual feature name
		self.get_feature_name()

		# get the pairing
		self.get_pairing_feature()

		# get grid shape
		self.get_grid_shape()

		# get the input shape
		self.get_input_shape()

		# get renormalization factor
		if self.normalize_features or self.normalize_targets:
			self.get_norm()


		print('\n')
		print("   Data Set Info")
		print('   Training set        : %d conformations' %self.ntrain)
		print('   Test set            : %d conformations' %(self.ntot-self.ntrain))
		print('   Number of channels  : %d' %self.input_shape[0])
		print('   Grid Size           : %d x %d x %d' %(self.data_shape[1],self.data_shape[2],self.data_shape[3]))
		sys.stdout.flush()



	def __len__(self):
		return len(self.index_complexes)


	def __getitem__(self,index):

		debug_time = False
		t0 = time.time()
		fname,mol = self.index_complexes[index]
		feature, target = self.load_one_molecule(fname,mol)
		printif('        __getitem__ : %f' %(time.time()-t0),debug_time)

		if self.normalize_features:
			feature = self._normalize_feature(feature)

		if self.normalize_targets:
			target = self._normalize_target(target)

		if self.transform:
			feature = self.convert2d(feature,self.proj2D)

		if self.pair_ind_feature:
			feature = self.make_feature_pair(feature,self.pair_indexes,self.pair_ind_feature)

		return feature,target


	def create_index_molecules(self):

		'''
		Create the indexing of each molecule in the dataset

		only create an indexing like
		[ ('1ak4.hdf5,1AK4_100w), 
		  ('1ak4.hdf5,1AK4_101w),
		....
		  ('1fqj.hdf5,1FGJ_399w),  
		  ('1fqj.hdf5,1FGJ_400w),  
		]
		so that with the index of each molecule
		we can find in which file it is stored
		and its group name in the file
		'''
		print("   Processing data set")

		self.index_complexes = []

		desc = '{:25s}'.format('   Train dataset')
		if self.tqdm:
			data_tqdm = tqdm(self.database,desc=desc,file=sys.stdout)
		else:
			print('   Train dataset')
			data_tqdm = self.database
		sys.stdout.flush()

		for fdata in data_tqdm:
			if self.tqdm:
				data_tqdm.set_postfix(mol=os.path.basename(fdata))
			try:
				fh5 = h5py.File(fdata,'r')
				mol_names = list(fh5.keys())
				self.index_complexes += [(fdata,k) for k in mol_names]
				fh5.close()
			except:
				print('\t\t-->Ignore File : '+fdata)

		self.ntrain = len(self.index_complexes)
		self.index_train = list(range(self.ntrain))

		if self.test_database is not None:
			
			desc = '{:25s}'.format('   Test dataset')
			if self.tqdm:
				data_tqdm = tqdm(self.test_database,desc=desc,file=sys.stdout)
			else:
				data_tqdm = self.test_database
				print('   Test dataset')
			sys.stdout.flush()

			for fdata in data_tqdm:
				if self.tqdm:
					data_tqdm.set_postfix(mol=os.path.basename(fdata))
				try:
					fh5 = h5py.File(fdata,'r')
					mol_names = list(fh5.keys())
					self.index_complexes += [(fdata,k) for k in mol_names]
					fh5.close()
				except:
					print('\t\t-->Ignore File : '+fdata)				

		self.ntot = len(self.index_complexes)
		self.index_test = list(range(self.ntrain,self.ntot))


	def get_feature_name(self):

		'''
		Create  the dictionarry with actual feature_type : [feature names]

		Add _chainA, _chainB to each feature names if we have individual storage
		create the dict if selec_features == 'all'
		create the dict if selec_features['XXX']  == 'all'
		'''

		# open a h5 file in case we need it
		f5 = h5py.File(self.database[0],'r')
		mol_name = list(f5.keys())[0]
		mapped_data = f5.get(mol_name + '/mapped_features/')

		# if we select all the features
		if self.select_feature == "all":

			# redefine dict	
			self.select_feature = {}

			# loop over the feat types and add all the feat_names 
			for feat_type,feat_names in mapped_data.items():
				self.select_feature[feat_type] = [name for name in feat_names]

		# if a selection was made
		else:

			# we loop over the input dict
			for feat_type,feat_names in self.select_feature.items():

				# if for a give type we need all the feature
				if feat_names == 'all':
					self.select_feature[feat_type] = [list(mapped_data[feat_type].keys())]

				# if we have stored the individual
				# chainA chainB data we need to expand the feature list
				elif '_ind' in feat_type:
					self.select_feature[feat_type] = list(chain.from_iterable((name+'_chainA',name+'_chainB') for name in feat_names))

		f5.close()


	def get_pairing_feature(self):

		if self.pair_ind_feature :

			self.pair_indexes = []
			start = 0
			for feat_type,feat_names in self.select_feature.items():
				nfeat = len(feat_names):
				if '_ind' in feat_type:
					self.pair_indexes += [ [i,i+1] for i in range(start,start+nfeat,2)]
				else:
					self.pair_indexes += [ [i] for i in range(start,start+nfeat)]
				start += n

	def get_input_shape(self):

		'''
		get the size of the data and input
		self.data_shape : shape of the raw 3d data set
		self.input_shape : input size of the CNN (potentially after 2d transformation)
		'''

		fname = self.database[0]
		feature,_ = self.load_one_molecule(fname)
		self.data_shape = feature.shape

		if self.pair_ind_feature:
			feature = self.make_feature_pair(feature,self.pair_indexes,self.pair_ind_feature)

		if self.transform:
			feature = self.convert2d(feature,self.proj2D)

		self.input_shape = feature.shape


	def get_grid_shape(self):

		'''
		Get the shape of the matrices
		'''

		fname = self.database[0]
		fh5 = h5py.File(fname,'r')
		mol = list(fh5.keys())[0]

		# get the mol
		mol_data = fh5.get(mol)

		# get the grid size
		if self.grid_shape is None:

			if 'grid_points' in mol_data:
				nx = mol_data['grid_points']['x'].shape[0]
				ny = mol_data['grid_points']['y'].shape[0]
				nz = mol_data['grid_points']['z'].shape[0]
				self.grid_shape = (nx,ny,nz)
		
			else:
				raise ValueError('Impossible to determine sparse grid shape.\n Specify argument grid_shape=(x,y,z)')

		fh5.close()

	
	def get_norm(self):

		print("   Normalization factor :")

		# declare the dict of class instance
		# where we'll store the normalization parameter
		self.param_norm = {'features':{},'targets':{}}
		for feat_type,feat_names in self.select_feature.items():
			self.param_norm['features'][feat_type] = {}
			for name in feat_names:
				self.param_norm['features'][feat_type][name] = NormParam()
		self.param_norm['targets'][self.select_target] = MinMaxParam()

		#try:
		self._read_norm()
		#except:
		#	print('   Could not load normalization data')
		#	self._compute_norm()

		# make array for fast access
		self.feature_mean,self.feature_std = [],[]
		for feat_type,feat_names in self.select_feature.items():
			for name in feat_names:
				self.feature_mean.append(self.param_norm['features'][feat_type][name].mean)
				self.feature_std.append(self.param_norm['features'][feat_type][name].std)

		self.target_min = self.param_norm['targets'][self.select_target].min
		self.target_max = self.param_norm['targets'][self.select_target].max

	def _read_norm(self):

		# loop through all the filename
		for f5 in self.database:

			# get the precalculated data
			fdata = os.path.splitext(f5)[0]+'_norm.pckl'

			# if the file doesn't exist we create it
			if not os.path.isfile(fdata):
				print("      Computing norm for ", f5)
				norm = NormalizeData(f5,shape=self.grid_shape)
				norm.get()

			# read the data
			data = pickle.load(open(fdata,'rb'))

			# handle the features
			for feat_type,feat_names in self.select_feature.items():
				for name in feat_names:
					mean = data['features'][feat_type][name].mean
					var = data['features'][feat_type][name].var
					self.param_norm['features'][feat_type][name].add(mean,var)

			# handle the target
			minv = data['targets'][self.select_target].min
			maxv = data['targets'][self.select_target].max
			self.param_norm['targets'][self.select_target].update(minv)
			self.param_norm['targets'][self.select_target].update(maxv)

		# process the std
		nfile = len(self.database)
		for feat_types,feat_dict in self.param_norm['features'].items():
			for feat in feat_dict:
				self.param_norm['features'][feat_types][feat].process(nfile)


	def _compute_norm(self):

		'''
		Get the normalization data from the entire data set
		This is used only if the .pckl files containing the
		normalization info for each .hdf5 files can't be found
		
		'''

		desc = '{:25s}'.format('   Normalization')
		if self.tqdm:
			data_tqdm = tqdm(self.index_train,desc=desc,file=sys.stdout)
		else:
			data_tqdm = self.index_train
			print('   Normalization')
		sys.stdout.flush()

		for index in data_tqdm:

			fname,mol = self.index_complexes[index]
			
			if self.tqdm:		
				data_tqdm.set_postfix(mol=os.path.basename(mol))

			# load the molecule
			feature, target = self.load_one_molecule(fname,mol)

			# target
			self.param_norm['targets'][self.select_target].update(target)

			# features
			ifeat = 0
			for feat_type,feat_names in self.select_feature.items():
				for name in feat_names:
					mean = np.mean(feature[ifeat])
					var = np.var(feature[ifeat])
					self.param_norm['features'][feat_type][name].add(mean,var)
					ifeat += 1


		# process the std
		nmol = len(self.index_train)
		for feat_types,feat_dict in self.param_norm['features'].items():
			for feat in feat_dict:
				self.param_norm['features'][feat_types][feat].process(nmol)



	def _normalize_target(self,target):

		target -= self.target_min
		target /= self.target_max
		return target

	def _normalize_feature(self,feature):
		# we convert the feature back to numpy
		# normlaize them as np array
		# and pass them back as torch tensor
		# that's faster than doing the processing in Torch .... 
		# 400 conf 
		#   -> 12 sec in numpy
		#   -> 18 sec in torch
		#feature = feature.numpy()
		for ic in range(self.data_shape[0]):
			feature[ic] = (feature[ic]-self.feature_mean[ic])/self.feature_std[ic]
		return feature
		#return FloatTensor(feature)


	def backtransform_target(self,data):
		data = FloatTensor(data)
		data *= self.target_max
		data += self.target_min
		return data.numpy()


	############################################
	# load the feature/target of a single molecule
	############################################
	def load_one_molecule(self,fname,mol=None):

		'''
		load the feature/target of a single molecule
		'''
		outtype = 'float32'
		fh5 = h5py.File(fname,'r')

		if mol is None:
			mol = list(fh5.keys())[0]

		# get the mol
		mol_data = fh5.get(mol)

		# get the features
		feature = []
		for feat_type,feat_names in self.select_feature.items():

			# see if the feature exists
			feat_dict = mol_data.get('mapped_features/'+feat_type)						
			
			# loop through all the desired feat names
			for name in feat_names:
				
				# extract the group
				data = feat_dict[name]

				# check its sparse attribute
				# if true get a FLAN
				# if flase direct import
				if data.attrs['sparse']:
					mat = sparse.FLANgrid(sparse=True,
						                  index=data['index'].value,
						                  value=data['value'].value,
						                  shape=self.grid_shape).to_dense()
				else:
					mat = data['value'].value

				# append to the list of features
				feature.append(mat)

		# get the target value
		target = mol_data.get('targets/'+self.select_target).value
		
		# close
		fh5.close()

		# make sure all the feature have exact same type
		# if they don't  collate_fn in the creation of the minibatch will fail. 
		# Note returning torch.FloatTensor makes each epoch twice longer ...
		return np.array(feature).astype(outtype),np.array([target]).astype(outtype)



	############################################
	# load the feature/target of a single molecule
	############################################
	def _old_load_one_molecule(self,fname,mol=None):

		'''
		load the feature/target of a single molecule
		- open the hdf5 file
		- get the molecule group
		- read the specified features
		- read the specified target
		- transform to 2d (optional)
		- close the hdf5
		- return the feature/target
		'''

		fh5 = h5py.File(fname,'r')
		if mol is None:
			mol = list(fh5.keys())[0]

		featgrp_name='mapped_features/'

		# get the mol
		mol_data = fh5.get(mol)
		if self.grid_shape is not None:
			shape = self.grid_shape
		elif 'grid_points' in mol_data:
			nx = mol_data['grid_points']['x'].shape[0]
			ny = mol_data['grid_points']['y'].shape[0]
			nz = mol_data['grid_points']['z'].shape[0]
			shape=(nx,ny,nz)
			self.grid_shape = shape
		
		else:
			raise ValueError('Impossible to determine sparse grid shape.\n Specify argument grid_shape=(x,y,z)')

		# load all the features
		if self.select_feature == 'all':
			# loop through the features
			feature = []
			for feat_name, feat_members in mol_data.get(featgrp_name).items():
				# loop through all the feature keys
				for name,data in feat_members.items():
			
					if data.attrs['sparse']:
						spg = sparse.FLANgrid(sparse=True,
							                 index=data['index'].value,
							                 value=data['value'].value,
							                 shape=shape)
						feature.append(spg.to_dense())
					else:
						feature.append(data['value'].value)
					

		# load selected features
		else:
			feature = []
			for feat_name,feat_channels in self.select_feature.items():

				# see if the feature exists
				feat_dict = mol_data.get(featgrp_name+feat_name)						
				
				if feat_dict is None:
					
					print('Error : Feature name %s not found in %s' %(feat_name,mol))
					opt_names = list(mol_data.get(featgrp_name).keys())
					print('Error : Possible features are \n\t%s' %'\n\t'.join(opt_names))
					sys.exit()

				# get the possible channels
				possible_channels = list(feat_dict.keys())

				# make sure that all the featchanels are in the file
				if feat_channels != 'all':
					for fc in feat_channels:
						if fc not in possible_channels:
							print("Error : required key %s for feature %s not in the database" %(fc,feat_name))
							print('Error : Possible features are \n\t%s' %'\n\t'.join(possible_channels))
							sys.exit()

				# load the feature channels
				for chanel_name,channel_data in feat_dict.items():
					if feat_channels == 'all' or chanel_name in feat_channels:
						if channel_data.attrs['sparse']:
							spg = sparse.FLANgrid(sparse=True,
								                  index=channel_data['index'].value,
								                  value=channel_data['value'].value,
								                  shape=shape)
							feature.append(spg.to_dense())
						else:
							feature.append(channel_data['value'].value)
						

			# append to the list of features
			feature = np.array(feature)

		# target
		opt_names = list(mol_data.get('targets/').keys())			
		fname = list(filter(lambda x: self.select_target in x, opt_names))
		

		if len(fname) == 0:
			print('Error : Target name %s not found in %s' %(self.select_target,mol))
			print('Error : Possible targets are \n\t%s' %'\n\t'.join(opt_names))
			sys.exit()

		if len(fname)>1:
			print('Error : Multiple Targets Matching %s Found in %s' %(self.select_target,mol))
			print('Error : Possible targets are \n\t%s' %'\n\t'.join(opt_names))
			sys.exit()

		fname = fname[0]
		target = mol_data.get('targets/'+fname).value
	
		# no TorchTensor transform
		feature = np.array(feature)
		target = np.array([target])
		
		# close
		fh5.close()

		return feature,target




	#convert the 3d data set to 2d data set
	@staticmethod
	def convert2d(feature,proj2d):

		'''
		convert the 3D volumetric feature to a 2D planar data set 
		to be used in 2d convolutional network
		proj2d specifies the dimension that we want to comsider as channel
		for example for proj2d = 0 the 2D images are in the yz plane and 
		the stack along the x dimension is considered as extra channels
		'''
		nc,nx,ny,nz = feature.shape
		if proj2d==0:
			feature = feature.reshape(-1,1,ny,nz).squeeze()
		elif proj2d==1:
			feature = feature.reshape(-1,nx,1,nz).squeeze()
		elif proj2d==2:
			feature = feature.reshape(-1,nx,ny,1).squeeze()
		
		return feature


	@staticmethod
	def make_feature_pair(feature,pair_indexes,op):

		if not callable(op):
			raise ValueError('Operation not callable',op)

		outtype = feature.dtype
		new_feat = []
		for ind in pair_indexes:
			if len(ind) == 1:
				new_feat.append(feature[ind,...])
			else:
				new_feat.append(op(feature[ind[0],...],feature[ind[1],...]))
		return np.array(new_feat).astype(outtype)

	