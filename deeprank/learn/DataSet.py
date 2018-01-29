import os
import sys
import time
import h5py
import torch
from torch import FloatTensor
import torch.utils.data as data_utils
import subprocess as sp
import numpy as np

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
		self.normalize_features = normalize_features
		self.normalize_targets = normalize_targets

		# final containers
		self.features = None
		self.targets  = None
		self.input_shape = None
		self.data_hape = None
		self.grid_shape = grid_shape

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

		# get the input shape
		self.get_shape()

		# get renormalization factor
		if self.normalize_features or self.normalize_targets:
			self.get_normalization()


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
		fname = self.index_complexes[index][0]
		mol = self.index_complexes[index][1]
		feature, target = self.load_one_molecule(fname,mol)
		printif('        __getitem__ : %f' %(time.time()-t0),debug_time)

		if self.normalize_targets:
			target = self._normalize_target(target)

		if self.normalize_features:
			feature = self._normalize_feature(feature)

		if self.transform:
			feature = self.convert2d(feature,self.proj2D)

		return feature,target

	########################################################
	# Create the indexing of each molecule in the dataset
	########################################################
	def create_index_molecules(self):

		'''
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
			fh5 = h5py.File(fdata,'r')
			mol_names = list(fh5.keys())
			self.index_complexes += [(fdata,k) for k in mol_names]
			fh5.close()

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
				fh5 = h5py.File(fdata,'r')
				mol_names = list(fh5.keys())
				self.index_complexes += [(fdata,k) for k in mol_names]
				fh5.close()

		self.ntot = len(self.index_complexes)
		self.index_test = list(range(self.ntrain,self.ntot))

	# get the input shape
	# That's useful to preprocess 
	# the model 
	def get_shape(self):

		'''
		get the size of the data and input
		the data is the raw 3d data set
		the input is the input size of the CNN (potentially after 2d transformation)
		'''

		fname = self.database[0]
		feature,_ = self.load_one_molecule(fname)
		self.data_shape = feature.shape

		if self.transform:
			feature = self.convert2d(feature,self.proj2D)
		self.input_shape = feature.shape

	# normalize the data
	# its a bit more complicated when you
	# load mol by mol
	# for the std see
	# https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
	# 
	def get_normalization(self):

		desc = '{:25s}'.format('   Normalization')
		if self.tqdm:
			data_tqdm = tqdm(range(self.__len__()),desc=desc,file=sys.stdout)
		else:
			data_tqdm = range(self.__len__())
			print('   Normalization')
		sys.stdout.flush()

		for index in data_tqdm:

			fname = self.index_complexes[index][0]
			mol = self.index_complexes[index][1]	

			if self.tqdm:		
				data_tqdm.set_postfix(mol=os.path.basename(mol))

			feature, target = self.load_one_molecule(fname,mol)

			# here we pass eveything in np format again
			# and then we re going to retransform  ..... 
			# dum dum dum
			#target = target.numpy()
			#feature = feature.numpy()

			if index == 0:

				self.target_min = torch.min(target)
				self.target_max = torch.max(target)

				#self.feature_mean = np.zeros(self.data_shape[0])
				#var  = np.zeros(self.data_shape[0])
				#sqmean = np.zeros(self.data_shape[0])

				self.feature_mean = FloatTensor(self.data_shape[0]).zero_()
				var = FloatTensor(self.data_shape[0]).zero_()
				sqmean = FloatTensor(self.data_shape[0]).zero_()
				

				for ic in range(self.data_shape[0]):
					self.feature_mean[ic] = torch.mean(feature[ic])
					var[ic] = torch.var(feature[ic])
					sqmean[ic] = self.feature_mean[ic]**2
			else:
				
				self.target_min = np.min(np.append(target.numpy(),self.target_min))
				self.target_max = np.max(np.append(target.numpy(),self.target_max))

				for ic in range(self.data_shape[0]):
					self.feature_mean[ic] += torch.mean(feature[ic])
					var[ic]  += torch.var(feature[ic])
					sqmean[ic] +=  torch.mean(feature[ic])**2

		# average the mean values
		self.feature_mean /= self.ntot
		
		# average the standard deviations
		self.feature_std = var/self.ntot
		self.feature_std += sqmean/self.ntot
		self.feature_std -= self.feature_mean**2
		self.feature_std = torch.sqrt(self.feature_std)

		# make torch tensor
		# takes quite a long time 
		#self.target_min = FloatTensor(np.array([self.target_min]))
		#self.target_max = FloatTensor(np.array([self.target_max]))
		#self.feature_mean = FloatTensor(self.feature_mean)
		#self.feature_std = FloatTensor(self.feature_std)

	def _normalize_target(self,target):

		target -= self.target_min
		target /= self.target_max
		return target

	def backtransform_target(self,data):
		data = FloatTensor(data)
		data *= self.target_max
		data += self.target_min
		return data.numpy()

	def _normalize_feature(self,feature):

		for ic in range(self.data_shape[0]):
			feature[ic] = (feature[ic]-self.feature_mean[ic])/self.feature_std[ic]
		return feature

	############################################
	# load the feature/target of a single molecule
	############################################
	def load_one_molecule(self,fname,mol=None):

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
	
		# transform the data in torch Tensor
		feature = FloatTensor(np.array(feature))
		target = FloatTensor(np.array([target]))

		# no TorchTensor transform
		#feature = np.array(feature)
		#target = np.array([target])
		
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



	