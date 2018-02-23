import os
import sys
import time
import h5py
import pickle

from torch import FloatTensor
import torch.utils.data as data_utils

import numpy as np

from deeprank.generate import NormParam,MinMaxParam,NormalizeData
from deeprank.tools import sparse

try:
	from tqdm import tqdm
except ImportError:
	def tqdm(x):
		return x

printif = lambda string,cond: print(string) if cond else None

class DataSet(data_utils.Dataset):

	'''
	Generates the data needed for deeprank.

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
		the name must correspond to a valid group : f[mol/targets/name]

	pair_chain_features

		numpy function to pair features of chainA and chainB
		example np.sum will add the data of the two chains

	dict_filter

		filter the complexes based on either
		- {name:[min,max]} the value of name must be between min max
		- {name:cond} the value must respect the condition e.g {'IRMSD':'>10 or <4'}

	trasnform_to_2d

		Boolean to use 2d maps instead of full 3d

	project

		Projection plane for the transformation to 2D
		0 -> yz, 1 -> xz, 2 -> xy

	grid_shape

		specify the shape of the grids in the HDF5 files
		Handy if one has removed the grid points from the data

	normalize_targets/normalize_features

		Specify if the targets/features must be normalized during the training

	tqdm

		Print the progress bar

	'''

	def __init__(self,database,test_database=None,
		         select_feature='all',select_target='DOCKQ',
		         pair_chain_feature = None,dict_filter = None,
		         transform_to_2D=False,projection=0,grid_shape = None,
		         normalize_features=True,normalize_targets=True,
		         clip_features=True,clip_factor=10.,
		         tqdm=False,process=True):


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

		# clip the data
		self.clip_features = clip_features
		self.clip_factor = clip_factor

		# final containers
		self.features = None
		self.targets  = None
		self.input_shape = None
		self.data_shape = None
		self.grid_shape = grid_shape

		# the possible pairing of the ind features
		self.pair_chain_feature = pair_chain_feature

		# get the eventual projection
		self.transform = transform_to_2D
		self.proj2D = projection

		# filter the dataset
		self.dict_filter = dict_filter

		# print the progress bar or not
		self.tqdm=tqdm

		# process the data
		if process:
			self.process_dataset()

	def process_dataset(self):

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


		# check if the files are ok
		self.check_hdf5_files()

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

		if self.clip_features:
			feature = self._clip_feature(feature)

		if self.normalize_features:
			feature = self._normalize_feature(feature)

		if self.normalize_targets:
			target = self._normalize_target(target)

		if self.transform:
			feature = self.convert2d(feature,self.proj2D)

		if self.pair_chain_feature:
			feature = self.make_feature_pair(feature,self.pair_indexes,self.pair_chain_feature)

		return {'mol':[fname,mol],'feature':feature,'target':target}
		#return feature,target


	def check_hdf5_files(self):

		print("   Checking dataset Integrity")
		remove_file = []
		for fname in self.database:
			try:
				f = h5py.File(fname)
				mol_names = list(f.keys())
				if len(mol_names) == 0:
					print('    -> %s is empty ' %fname)
					remove_file.append(fname)
				f.close()
			except:
				print('    -> %s is corrputed ' %fname)
				remove_file.append(fname)

		for name in remove_file:
			self.database.remove(name)


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
				for k in mol_names:
					if self.filter(fh5[k]):
						self.index_complexes += [(fdata,k)]
				fh5.close()
			except Exception as inst:
				print('\t\t-->Ignore File : ' + fdata)
				print(inst)

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


	def filter(self,molgrp):
		'''
		Filter the molecule according to a dictionary
		The dictionary must be of the form: { 'name' : [minv,maxv] }
		or:                                 None
		if None : no conditions are applied
		if{'name':[min,max]}
		'name' must be a valid dataset in fh5[mol/targets/]
		minv/maxv can be None and are then replaced by -/+ Inf
		Return True if no condition were provided or if all the test are passed
		Return False otherwise
		'''
		if self.dict_filter is None:
			return True

		for cond_name,cond_vals in self.dict_filter.items():

			try:
				val = molgrp['targets/'+cond_name].value
			except KeyError:
				print('   :Filter %s not found for mol %s' %(cond_name,mol))

			# if we have a list we assume that we want
			# the value to be between the bound
			if isinstance(cond_vals,list):

				minv = -float('Inf') if cond_vals[0] is None else cond_vals[0]
				maxv =  float('Inf') if cond_vals[1] is None else cond_vals[1]

				if val < minv or val > maxv:
					return False

			# if we have a string it's more complicated
			elif isinstance(cond_vals,str):

				ops = ['>','<','==']
				new_cond_vals = cond_vals
				for o in ops:
					new_cond_vals = new_cond_vals.replace(o,'val'+o)
				if not eval(new_cond_vals):
					return False

		return True

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
		chain_tags = ['_chainA','_chainB']

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
					self.select_feature[feat_type] = list(mapped_data[feat_type].keys())

				# if we have stored the individual
				# chainA chainB data we need to expand the feature list
				# however when we reload we already come with _chainA, _chainB features
				# so then we shouldn't add the tags
				elif '_ind' in feat_type:
					self.select_feature[feat_type] = []
					for name in feat_names:
						cond = [tag not in name for tag in chain_tags]
						if np.all(cond):
							self.select_feature[feat_type] += [name+tag for tag in chain_tags]
						else:
							self.select_feature[feat_type].append(name)
		f5.close()


	def get_pairing_feature(self):

		if self.pair_chain_feature:

			self.pair_indexes = []
			start = 0
			for feat_type,feat_names in self.select_feature.items():
				nfeat = len(feat_names)
				if '_ind' in feat_type:
					self.pair_indexes += [ [i,i+1] for i in range(start,start+nfeat,2)]
				else:
					self.pair_indexes += [ [i] for i in range(start,start+nfeat)]
				start += nfeat

	def get_input_shape(self):

		'''
		get the size of the data and input
		self.data_shape : shape of the raw 3d data set
		self.input_shape : input size of the CNN (potentially after 2d transformation)
		'''

		fname = self.database[0]
		feature,_ = self.load_one_molecule(fname)
		self.data_shape = feature.shape

		if self.pair_chain_feature:
			feature = self.make_feature_pair(feature,self.pair_indexes,self.pair_chain_feature)

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

		# read the normalization
		self._read_norm()

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
					if var == 0:
						print('  : STD is null for %s in %s' %(name,f5))
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
				if self.param_norm['features'][feat_types][feat].std == 0:
					print('  Final STD Null for %s/%s. Changed it to 1' %(feat_types,feat))
					self.param_norm['features'][feat_types][feat].std = 1

	def backtransform_target(self,data):
		data = FloatTensor(data)
		data *= self.target_max
		data += self.target_min
		return data.numpy()

	def _normalize_target(self,target):

		target -= self.target_min
		target /= self.target_max
		return target

	def _normalize_feature(self,feature):
		for ic in range(self.data_shape[0]):
			feature[ic] = (feature[ic]-self.feature_mean[ic])/self.feature_std[ic]
		return feature

	def _clip_feature(self,feature):
		w = self.clip_factor
		for ic in range(self.data_shape[0]):
			minv = self.feature_mean[ic] - w*self.feature_std[ic]
			maxv = self.feature_mean[ic] + w*self.feature_std[ic]
			feature[ic] = np.clip(feature[ic],minv,maxv)
		return feature


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
			print(ind)
			if len(ind) == 1:
				new_feat.append(feature[ind,...])
			else:
				new_feat.append(op(feature[ind[0],...],feature[ind[1],...]))

		return np.array(new_feat).astype(outtype)

