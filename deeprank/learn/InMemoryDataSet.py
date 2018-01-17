import os
import sys
import time
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

printif = lambda string,cond: print(string) if cond else None

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

	def __init__(self,database,test_database=None,
		         select_feature='all',select_target='DOCKQ',
		         transform_to_2D=False,projection=0,grid_shape=None,
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
		self.data_shape = None
		self.grid_shape = grid_shape

		# get the eventual projection
		self.transform = transform_to_2D
		self.proj2D = projection

		# print the progress bar or not
		self.tqdm = tqdm

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


		# load the dataset
		self._load()


		print('\n')
		print("   Data Set Info")
		print('   Training set        : %d conformations' %self.ntrain)
		print('   Test set            : %d conformations' %(self.ntot-self.ntrain))
		print('   Number of channels  : %d' %self.input_shape[0])
		print('   Grid Size           : %d x %d x %d' %(self.data_shape[1],self.data_shape[2],self.data_shape[3]))


	def __len__(self):
		return len(self.targets)

	def __getitem__(self,index):
		debug_time = False
		t0 = time.time()
		f,t = self.features[index],self.targets[index]
		printif('        __getitem__ : %f' %(time.time()-t0),debug_time)
		return f,t

	# load the dataset from a h5py file
	def _load(self):

		lendata = [0,0]
		features, targets = [], []
		if self.test_database is None:
			iter_databases = [self.database]
		else:
			iter_databases = [self.database,self.test_database]
		for idata,database in enumerate(iter_databases):
			
			for fdata in database:

				fh5 = h5py.File(fdata,'r')
				mol_names = list(fh5.keys())
				featgrp_name='mapped_features/'
				lendata[idata] += len(mol_names)

				#	
				# load the data
				# the format of the features must be
				# Nconf x Nchanels x Nx x Ny x Nz
				# 
				# for each folder we create a tmp_feat of size Nchanels x Nx x Ny x Nz
				# that we then append to the feature list
				#
				
				for mol in tqdm(mol_names):

					# get the mol
					mol_data = fh5.get(mol)
					if self.grid_shape is not None:
						shape = self.grid_shape
					elif 'grid_points' in mol_data:
						nx = mol_data['grid_points']['x'].shape[0]
						ny = mol_data['grid_points']['y'].shape[0]
						nz = mol_data['grid_points']['z'].shape[0]
						shape=(nx,ny,nz)
					else:
						raise ValueError('Impossible to determine sparse grid shape.\n Specify argument grid_shape=(x,y,z)')

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

				# close
				fh5.close()

		# preprocess the data
		self.preprocess(features,targets)

		self.ntrain = lendata[0]
		self.ntot = lendata[0]+lendata[1]
		self.index_train = list(range(self.ntrain))
		self.index_test = list(range(self.ntrain,self.ntot))

	# put in torch format and normalize
	def preprocess(self,features,targets):

		# get the number of channels and points along each axis
		self.input_shape = features[0].shape
		self.data_shape = features[0].shape

		# transform the data in torch Tensor
		self.features = FloatTensor(np.array(features))
		self.targets = FloatTensor(np.array(targets))

		# normalize the targets/featurs
		if self.normalize_targets:
			self.target_max = self.targets.max()
			self.target_min = self.targets.min()
			self.targets -= self.target_min
			self.targets /= self.target_max

		if self.normalize_features:
			for iC in range(self.features.shape[1]):
				self.features[:,iC,:,:,:] = (self.features[:,iC,:,:,:]-self.features[:,iC,:,:,:].mean())/self.features[:,iC,:,:,:].std()


	def backtransform_target(self,data):
		data = FloatTensor(data)
		data *= self.target_max
		data += self.target_min
		return data.numpy()


	#convert the 3d data set to 2d data set
	def convert_dataset_to2d(self):

		'''
		convert the 3D volumetric dataset to a 2D planar data set 
		to be used in 2d convolutional network
		proj2d specifies the dimension that we want to comsider as channel
		for example for proj2d = 0 the 2D images are in the yz plane and 
		the stack along the x dimension is considered as extra channels
		'''
		planes = ['yz','xz','xy']
		print(': Project 3D data set to 2D images in the %s plane ' %planes[self.proj2D])
		self.data_shape = self.input_shape
		nf = self.__len__()
		nc,nx,ny,nz = self.input_shape
		if self.proj2D==0:
			self.features = self.features.view(nf,-1,1,ny,nz).squeeze()
		elif self.proj2D==1:
			self.features = self.features.view(nf,-1,nx,1,nz).squeeze()
		elif self.proj2D==2:
			self.features = self.features.view(nf,-1,nx,ny,1).squeeze()
		
		# get the number of channels and points along each axis
		# the input_shape is now a torch.Size object
		self.input_shape = self.features[0].shape
