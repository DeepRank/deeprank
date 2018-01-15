import os
import sys
import h5py
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

	def __init__(self,database,test_database=None,select_feature='all',select_target='DOCKQ',
		         transform_to_2D=False,projection=0,grid_shape = None,normalize_features=True,normalize_targets=True):


		self.database = database
		self.test_database = test_database
		self.select_feature = select_feature
		self.select_target = select_target
		self.grid_shape = grid_shape
		self.features = None
		self.targets  = None

		self.normalize_features = normalize_features
		self.normalize_targets = normalize_targets

		# allow for multiple database
		if not isinstance(database,list):
			self.database = [database]

		# allow for multiple database
		if test_database is not None:
			if not isinstance(test_database,list):
				self.test_database = [test_database]

		# get the eventual projection
		self.transform = transform_to_2D
		self.proj2D = projection

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

		# create the indexing system
		# alows to associate each mol to an index
		# and get fname and mol name from the index
		self.create_index_molecules()

		# get the input shape
		self.get_input_shape()

		# get renormalization factor
		if self.normalize_features or self.normalize_targets:
			self.get_normalization()

		print('\n')
		print("   Data Set Info")
		print('   Training set        : %d conformations' %self.ntrain)
		print('   Test set            : %d conformations' %(self.ntot-self.ntrain))
		print('   Number of channels  : %d' %self.input_shape[0])
		print('   Grid Size           : %d x %d x %d' %(self.input_shape[1],self.input_shape[2],self.input_shape[3]))


	def __len__(self):
		return len(self.index_complexes)

	def __getitem__(self,index):

		fname = self.index_complexes[index][0]
		mol = self.index_complexes[index][1]
		
		feature, target = self.load_one_molecule(fname,mol)

		if self.normalize_targets:
			target = self._normalize_target(target)

		if self.normalize_features:
			feature = self._normalize_feature(feature)

		if self.transform:
			feature = self.convert2d(feature,self.proj2d)

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
		data_tqdm = tqdm(self.database,desc=desc)
		for fdata in data_tqdm:
			data_tqdm.set_postfix(mol=os.path.basename(fdata))
			fh5 = h5py.File(fdata,'r')
			mol_names = list(fh5.keys())
			self.index_complexes += [(fdata,k) for k in mol_names]
			fh5.close()

		self.ntrain = len(self.index_complexes)
		self.index_train = list(range(self.ntrain))

		if self.test_database is not None:
			
			desc = '{:25s}'.format('   Test dataset')
			data_tqdm = tqdm(self.test_database,desc=desc)

			for fdata in data_tqdm:
				data_tqdm.set_postfix(mol=os.path.basename(fdata))
				fh5 = h5py.File(fdata,'r')
				mol_names = list(fh5.keys())
				self.index_complexes += [(fdata,k) for k in mol_names]
				fh5.close()

		self.ntot = len(self.index_complexes)
		self.index_valid = list(range(self.ntrain,self.ntot))

	# get the input shape
	# That's useful to preprocess 
	# the model 
	def get_input_shape(self):

		fname = self.database[0]
		feature,_ = self.load_one_molecule(fname)
		self.input_shape = feature.shape

	# normalize the data
	# its a bit more complicated when you
	# load mol by mol
	# for the std see
	# https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
	# 
	def get_normalization(self):

		desc = '{:25s}'.format('   Normalization')
		data_tqdm = tqdm(range(self.__len__()),desc=desc)
		for index in data_tqdm:

			fname = self.index_complexes[index][0]
			mol = self.index_complexes[index][1]			
			data_tqdm.set_postfix(mol=os.path.basename(mol))

			feature, target = self.load_one_molecule(fname,mol)

			target = target.numpy()
			feature = feature.numpy()

			if index == 0:

				self.target_min = np.min(target)
				self.target_max = np.min(target)

				self.feat_mean = np.zeros(self.input_shape[0])

				var  = np.zeros(self.input_shape[0])
				sqmean = np.zeros(self.input_shape[0])

				for ic in range(self.input_shape[0]):
					self.feat_mean[ic] = feature[ic].mean()
					var[ic] = feature[ic].var()
					sqmean[ic] = feature[ic].mean()**2
			else:
				
				self.target_min = np.min([self.target_min,target])
				self.target_max = np.max([self.target_max,target])

				for ic in range(self.input_shape[0]):
					self.feat_mean[ic] += feature[ic].mean()
					var[ic]  += feature[ic].var()
					sqmean[ic] += feature[ic].mean()**2

		# average the mean values
		self.feat_mean /= self.ntot
		
		# average the standard deviations
		self.feat_std = var/self.ntot
		self.feat_std += sqmean/self.ntot
		self.feat_std -= self.feat_mean**2
		self.feat_std = np.sqrt(self.feat_std)

		# make torch tensor
		self.target_min = FloatTensor(np.array([self.target_min]))
		self.target_max = FloatTensor(np.array([self.target_max]))
		self.feat_mean = FloatTensor(self.feat_mean)
		self.feat_std = FloatTensor(self.feat_std)

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

		for ic in range(self.input_shape[0]):
			feature[ic] = (feature[ic]-self.feat_mean[ic])/self.feat_std[ic]
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
		if 'grid_points' in mol_data:
			nx = mol_data['grid_points']['x'].shape[0]
			ny = mol_data['grid_points']['y'].shape[0]
			nz = mol_data['grid_points']['z'].shape[0]
			shape=(nx,ny,nz)
		elif self.grid_shape is not None:
			shape = self.grid_shape
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
			feature = feature.view(-1,1,ny,nz).squeeze()
		elif proj2d==1:
			feature = feature.view(-1,nx,1,nz).squeeze()
		elif proj2d==2:
			feature = feature.view(-1,nx,ny,1).squeeze()
		
		return feature

class InMemoryDataSet(data_utils.Dataset):

	'''
	Class that generates the data needed for deeprank.

	All the data is stored in memory. That allows fast access to 
	the data but limit the dataset to small size.

	For bigger dataset use Dataset()

	ARGUMENTS 
		
	database : string or list of strings

		Path of the HDF5 file(s) containing the database(s)
		This hdf5 file(s) must be generated with the deeprank.generate
		tools to insure the correct structure


	filter_dataset : None, integer or file name

		Int
		Maximum number of elements required in the data set.
		It might be that the data_folder contains N conformation
		but that we only want M of them. Then we can set dataset_size = M
		In that case we select M random comformation among the N

		File Name
		the file name must contains complex ID that are in the data_folder
		the data set will be only constucted from the cmplexes specified in
		the file.

		None (default value)
		all the data are loaded

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

	normalize_x   :	Boolean

		normalize the values of the features/targets between 0 and 1


	USAGE

		data = DeepRankDataSet(fname)

	'''

	def __init__(self,database,filter_dataset=None,
				 select_feature='all',select_target='DOCKQ',
		         normalize_features=True,normalize_targets=True):


		self.database = database
		self.filter_dataset = filter_dataset
		self.select_feature = select_feature
		self.select_target = select_target
		self.normalize_features = normalize_features
		self.normalize_targets = normalize_targets

		self.features = None
		self.targets  = None

		self.input_shape = None

		# allow for multiple database
		if not isinstance(database,list):
			self.database = [database]

		# directly load the dataset
		self._load()

	def __len__(self):
		return len(self.targets)

	def __getitem__(self,index):
		return self.features[index],self.targets[index]

	# load the dataset from a h5py file
	def _load(self):

		for fdata in self.database:

			fh5 = h5py.File(fdata,'r')
			mol_names = list(fh5.keys())
			featgrp_name='mapped_features/'

			# get a subset
			if self.filter_dataset != None:

				# get a random subset if integers
				if isinstance(self.filter_dataset,int):
					np.random.shuffle(mol_names)
					mol_names = mol_names[:self.filter_dataset]

				# select based on name of file
				if os.path.isfile(self.filter_dataset):
					tmp_folder = []
					with open(self.filter_dataset) as f:
						for line in f:
							if len(line.split())>0:
								name = line.split()[0]
								tmp_folder += list(filter(lambda x: name in x,mol_names))
					f.close()
					mol_names = tmp_folder

			#	
			# load the data
			# the format of the features must be
			# Nconf x Nchanels x Nx x Ny x Nz
			# 
			# for each folder we create a tmp_feat of size Nchanels x Nx x Ny x Nz
			# that we then append to the feature list
			#
			features, targets = [], []
			for mol in tqdm(mol_names):

				# get the mol
				mol_data = fh5.get(mol)
				nx = mol_data['grid_points']['x'].shape[0]
				ny = mol_data['grid_points']['y'].shape[0]
				nz = mol_data['grid_points']['z'].shape[0]
				shape=(nx,ny,nz)

				# load all the features
				if self.select_feature == 'all':
					# loop through the features
					tmp_feat = []
					for feat_name, feat_members in mol_data.get(featgrp_name).items():
						# loop through all the feature keys
						for name,data in feat_members.items():
							if data.attrs['sparse']:
								spg = sparse.FLANgrid(sparse=True,
									                 index=data['index'].value,
									                 value=data['value'].value,
									                 shape=shape)
								tmp_feat.append(spg.to_dense())
							else:
								tmp_feat.append(data['value'].value)
					features.append(np.array(tmp_feat))

				# load selected features
				else:
					tmp_feat = []
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
									tmp_feat.append(spg.to_dense())
								else:
									tmp_feat.append(channel_data['value'].value)
								

					# append to the list of features
					features.append(np.array(tmp_feat))


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
				targ_data = mol_data.get('targets/'+fname)		
				targets.append(targ_data.value)

			# preprocess the data
			self.preprocess(features,targets)

			# close
			fh5.close()

	# put in torch format and normalize
	def preprocess(self,features,targets):

		# get the number of channels and points along each axis
		self.input_shape = features[0].shape

		# transform the data in torch Tensor
		self.features = FloatTensor(np.array(features))
		self.targets = FloatTensor(np.array(targets))

		# normalize the targets/featurs
		if self.normalize_targets:
			self.targets -= self.targets.min()
			self.targets /= self.targets.max()

		if self.normalize_features:
			for iC in range(self.features.shape[1]):
				self.features[:,iC,:,:,:] = (self.features[:,iC,:,:,:]-self.features[:,iC,:,:,:].mean())/self.features[:,iC,:,:,:].std()

	#convert the 3d data set to 2d data set
	def convert_dataset_to2d(self,proj2d=0):

		'''
		convert the 3D volumetric dataset to a 2D planar data set 
		to be used in 2d convolutional network
		proj2d specifies the dimension that we want to comsider as channel
		for example for proj2d = 0 the 2D images are in the yz plane and 
		the stack along the x dimension is considered as extra channels
		'''
		print(': Project 3D data set to 2D images in the %s plane ' %planes[proj2d])

		nf = self.__len__()
		nc,nx,ny,nz = self.input_shape
		if proj2d==0:
			self.features = self.features.view(nf,-1,1,ny,nz).squeeze()
		elif proj2d==1:
			self.features = self.features.view(nf,-1,nx,1,nz).squeeze()
		elif proj2d==2:
			self.features = self.features.view(nf,-1,nx,ny,1).squeeze()
		
		# get the number of channels and points along each axis
		# the input_shape is now a torch.Size object
		self.input_shape = self.features[0].shape