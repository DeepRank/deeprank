import numpy as np
import os
import h5py 
import pickle 
from deeprank.tools import sparse

class NormParam(object):

	'''
	Normalization data to get the mean,var,std of all
	the data in the hdf5 file
	The standard deviation of a given channel is calculated from the std of all the individual grids
	This is done following:
	https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation


	\sigma_{tot} = \sqrt{ \frac{1}{N} \Sum_i \sigma_i^2 + \frac{1}{N} \sum_i \mu_i^2 - (\frac{1}{N} \Sum_i \mu_i  )^2 }

	'''

	def __init__(self,std=0,mean=0,var=0,sqmean=0):
		self.std = std
		self.mean = mean
		self.var = var
		self.sqmean = sqmean
		

	def add(self,mean,var):
		self.mean += mean
		self.sqmean += mean**2
		self.var += var

	def process(self,n):

		# normalize the mean and var
		self.mean   /= n
		self.var    /= n
		self.sqmean /= n 

		# get the std
		self.std = self.var
		self.std += self.sqmean
		self.std -= self.mean**2
		self.std = np.sqrt(self.std)

class MinMaxParam(object):

	def __init__(self,minv=None,maxv=None):
		self.min = minv
		self.max = maxv

	def update(self,val):

		if self.min is None:
			self.min = val
			self.max = val
		else:
			self.min = min(self.min,val)
			self.max = max(self.max,val)


class NormalizeData(object):

	def __init__(self,fname,shape=None):

		self.fname = fname
		self.parameters = {'features':{},'targets':{}}
		self.shape = shape

	def get(self):

		self.extract_shape()
		self.extract_data()
		self.process_data()
		self.export_data()

	def extract_shape(self):

		if self.shape is not None:
			return

		f5 = h5py.File(self.fname,'r')
		mol = list(f5.keys())[0]		
		mol_data = f5.get(mol)

		if 'grid_points' in mol_data:

			nx = mol_data['grid_points']['x'].shape[0]
			ny = mol_data['grid_points']['y'].shape[0]
			nz = mol_data['grid_points']['z'].shape[0]
			self.shape=(nx,ny,nz)

		else:
			raise ValueError('Impossible to determine sparse grid shape.\n Specify argument grid_shape=(x,y,z)')	

	def extract_data(self):

		f5 = h5py.File(self.fname,'r')
		mol_names = list(f5.keys())
		self.nmol = len(mol_names)

		# loop over the molecules
		for mol in mol_names:

			#get the mapped features group
			data_group = f5.get(mol+'/mapped_features/')

			# loop over all the feature types 
			for feat_types,feat_names in data_group.items():

				# if feature type not in param add
				if feat_types not in self.parameters['features']:
					self.parameters['features'][feat_types] = {}

				# loop over all the feature
				for name in feat_names:

					# create the param if it doens exists
					if name not in self.parameters['features'][feat_types]:
						self.parameters['features'][feat_types][name] = NormParam()

					# load the matrix
					feat_data = data_group[feat_types+'/'+name]
					if feat_data.attrs['sparse']:
						mat = sparse.FLANgrid(sparse=True,
							                  index=feat_data['index'].value,
							                  value=feat_data['value'].value,
							                  shape=self.shape).to_dense()
					else:
						mat = feat_data['value'].value

					# add the parameter (mean and var)
					self.parameters['features'][feat_types][name].add(np.mean(mat),np.var(mat))

			# get the target groups
			target_group = f5.get(mol+'/targets')

			# loop over all the targets
			for tname,tval in target_group.items():

				# create a new item if needed
				if tname not in self.parameters['targets']:
					self.parameters['targets'][tname] = MinMaxParam()

				# update the value
				self.parameters['targets'][tname].update(tval.value)

		f5.close()

	def process_data(self):
		for feat_types,feat_dict in self.parameters['features'].items():
			for feat in feat_dict:
				self.parameters['features'][feat_types][feat].process(self.nmol)

	def export_data(self):
		fexport = os.path.splitext(self.fname)[0] + '_norm.pckl'
		pickle.dump(self.parameters,open(fexport,'wb'))
