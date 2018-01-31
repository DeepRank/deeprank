import numpy as np
import subprocess as sp
import os, sys 
import itertools
from scipy.signal import bspline
import scipy.sparse as spsp
from collections import OrderedDict
from time import time
import logging

from deeprank.tools import pdb2sql
from deeprank.tools import sparse
try:
	from tqdm import tqdm
except:
	def tqdm(x):
		return x 

printif = lambda string,cond: print(string) if cond else None

# the main gridtool class
class GridTools(object):

	'''
	
	Map the feature of a complex on the grid


	ARGUMENTS

	molgrp

			the name of the group of the molecule in the HDF5 file
	          
	hdf5_file

			the file handler of th HDF5 file where to store the grids

	number_of_points 

			the number of points we want in each direction of the grid

	resolution

			the distance (in Angs) between two points we want. 

	atomic_densities

			dictionary of atom types cand their vdw radius
			exemple {'CA':3.5, 'CB':3.0}
			The correspondign atomic densities will be mapped on the grid 
			and exported to the hdf5 file

	feature

			Name of the features to be mapped. By default all the features
			present in hdf5_file['<molgrp>/features/] will be mapped

	atomic_densities_mode
	feature_mode

			The mode for mapping
			'sum'  --> chainA + chainB
			'diff' --> chainA - chainB
			'ind'  --> chainA and chainB in separate grids

	contact distance
			the dmaximum distance between two contact atoms default 8.5 A

	
	cuda
			Use CUDA or not 

	gpu_block
			GPU block size to be used e.g. [8,8,8]


	cuda_func
			Name of the CUDA function to be used for the mapping 
			of the features. Must be present in kernel_cuda.c

	cuda_atomic
			Name of the CUDA function to be used for the mapping 
			of the atomic densities. Must be present in kernel_cuda.c		

	prog_bar
			print progression bar for individual grid (default False)

	time
			print timimg statistic for individual grid (default False)

	try_sparse
			Try to store the matrix in sparse format (default True)

	logger
			logger 


	USAGE


	grid = GridTools(mogrp='1AK4_1w',hdf5_file=fhandle
		             atomic_densities={'CA':3.5},
		             number_of_points = [30,30,30],
		             resolution = [1.,1.,1.])


	OUTPUT : all files are stored in the HDF5 file
	
	'''

	def __init__(self, molgrp,
				number_of_points = [30,30,30],resolution = [1.,1.,1.],
				atomic_densities=None, atomic_densities_mode='ind',
				feature = None, feature_mode  ='ind',
				contact_distance = 8.5, hdf5_file=None,
				cuda=False, gpu_block=None, cuda_func=None, cuda_atomic=None,
				prog_bar = False,time=False,try_sparse=True,logger=None):
		
		# mol file	
		self.molgrp = molgrp

		# feature requestedOO
		self.atomic_densities = atomic_densities
		self.feature = feature

		# mapping mode
		self.feature_mode = feature_mode
		self.atomic_densities_mode = atomic_densities_mode

		# base name of the molecule
		self.mol_basename = molgrp.name
		
		# hdf5 file
		self.hdf5 = hdf5_file
		self.try_sparse = try_sparse

		# export to HDF5 file
		self.hdf5.require_group(self.mol_basename+'/features/')

		# parameter of the grid
		if number_of_points is not None:
			self.npts = np.array(number_of_points).astype('int')

		if resolution is not None:
			self.res  = np.array(resolution)

		# cuda support 
		self.cuda = cuda
		if self.cuda:
			self.gpu_block = gpu_block
			self.gpu_grid = [ int(np.ceil(n/b)) for b,n in zip(self.gpu_block,self.npts)]
			

		# parameter of the atomic system
		self.atom_xyz = None
		self.atom_index = None
		self.atom_type = None

		# grid points
		self.x = None
		self.y = None
		self.z = None

		# cuda 
		self.cuda_func = cuda_func
		self.cuda_atomic = cuda_atomic

		# grids for calculation of atomic densities
		self.xgrid = None
		self.ygrid = None
		self.zgrid = None 

		# dictionaries of atomic densities
		self.atdens = {}

		# conversion from boh to angs for VMD visualization
		self.bohr2ang = 0.52918

		# contact distance to locate the interface
		self.contact_distance = contact_distance

		# progress bar
		self.local_tqdm = lambda x: tqdm(x) if prog_bar else x
		self.time = time
		self.logger = logger or logging.getLogger(__name__)

		# if we already have an output containing the grid
		# we update the existing features
		_update_ = False	
		if self.mol_basename+'/grid_points' in self.hdf5:
			_update_ = True

		if _update_:
			printif('\n= Updating grid data for %s' %(self.mol_basename),self.time)
			self.update_feature()
		else:
			printif('\n= Creating grid and grid data for %s' %(self.mol_basename),self.time)	
			self.create_new_data()
		



	################################################################

	def create_new_data(self):

		# get the position/atom type .. of the complex
		self.read_pdb()

		#get the contact atoms
		self.get_contact_atoms()

		# define the grid 
		self.define_grid_points()

		# save the grid points
		self.export_grid_points()

		#map the features
		self.add_all_features()

		# if we wnat the atomic densisties
		self.add_all_atomic_densities()

		# cloe the db file
		self.sqldb.close()

	################################################################

	def update_feature(self):

		# get the position/atom type .. of the complex
		self.read_pdb()

		# read the grid from the hdf5
		if self.hdf5 is not None:
			grid = self.hdf5.get(self.mol_basename+'/grid_points/')
			self.x,self.y,self.z = grid['x'][()],grid['y'][()],grid['z'][()]

		# or read the grid points from file
		else:
			grid = np.load(self.export_path+'/grid_points.npz')
			self.x,self.y,self.z = grid['x'], grid['y'], grid['z']

		# create the grid
		self.ygrid,self.xgrid,self.zgrid = np.meshgrid(self.y,self.x,self.z)
		
		# set the resolution/dimension
		self.npts = np.array([len(self.x),len(self.y),len(self.z)])
		self.res = np.array([self.x[1]-self.x[0],self.y[1]-self.y[0],self.z[1]-self.z[0]])

		# map the features
		self.add_all_features()

		# if we want the atomic densisties
		self.add_all_atomic_densities()			

		# cloe the db file
		self.sqldb.close()

	################################################################


	def read_pdb(self):
		self.sqldb = pdb2sql(self.molgrp['complex'].value)


	# get the contact atoms
	def get_contact_atoms(self):

		xyz1 = np.array(self.sqldb.get('x,y,z',chainID='A'))
		xyz2 = np.array(self.sqldb.get('x,y,z',chainID='B'))

		index_b = self.sqldb.get('rowID',chainID='B')

		self.contact_atoms = []
		for i,x0 in enumerate(xyz1):
			contacts = np.where(np.sqrt(np.sum((xyz2-x0)**2,1)) < self.contact_distance)[0]

			if len(contacts) > 0:
				self.contact_atoms += [i]
				self.contact_atoms += [index_b[k] for k in contacts]

		# create a set of unique indexes
		self.contact_atoms = list(set(self.contact_atoms))

		# get the mean xyz position
		self.center_contact = np.mean(np.array(self.sqldb.get('x,y,z',rowID=self.contact_atoms)),0)



	################################################################
	# shortcut to add all the feature a
	# and atomic densities in just one line
	################################################################

	# add all the residue features to the data
	def add_all_features(self):

		#map the features
		if self.feature is not None:
		
			# map the residue features
			dict_data = self.map_features(self.feature)

			# save to hdf5 if specfied
			t0 = time()
			printif('-- Save Features to HDF5',self.time)
			self.hdf5_grid_data(dict_data,'Feature_%s' %( self.feature_mode))
			printif('      Total %f ms' %((time()-t0)*1000),self.time)
		
	

	# add all the atomic densities to the data
	def add_all_atomic_densities(self):

		# if we wnat the atomic densisties
		if self.atomic_densities is not None:

			# compute the atomic densities
			self.map_atomic_densities()

			# save to hdf5
			t0 = time()
			printif('-- Save Atomic Densities to HDF5',self.time)
			self.hdf5_grid_data(self.atdens,'AtomicDensities_%s' %(self.atomic_densities_mode))
			printif('      Total %f ms' %((time()-t0)*1000),self.time)

			

	################################################################
	# define the grid points
	# there is an issue maybe with the ordering
	# In order to visualize the data in VMD the Y and X axis must be inverted ... 
	# I keep it like that for now as it should not matter for the CNN
	# and maybe we don't need atomic denisties as features
	################################################################

	def define_grid_points(self):

		printif('-- Define %dx%dx%d grid ' %(self.npts[0],self.npts[1],self.npts[2]),self.time)
		printif('-- Resolution of %1.2fx%1.2fx%1.2f Angs' %(self.res[0],self.res[1],self.res[2]),self.time)


		halfdim = 0.5*(self.npts*self.res)
		center = self.center_contact

		low_lim = center-halfdim
		hgh_lim = low_lim + self.res*(np.array(self.npts)-1)

		self.x = np.linspace(low_lim[0],hgh_lim[0],self.npts[0])
		self.y = np.linspace(low_lim[1],hgh_lim[1],self.npts[1])
		self.z = np.linspace(low_lim[2],hgh_lim[2],self.npts[2])


		# there is something fishy about the meshgrid 3d
		# the axis are a bit screwy .... 
		# i dont quite get why the ordering is like that 
		self.ygrid,self.xgrid,self.zgrid = np.meshgrid(self.y,self.x,self.z)

	################################################################
	# Atomic densities
	# as defined in the paper about ligand in protein
	################################################################

	# compute all the atomic densities data
	def map_atomic_densities(self,only_contact=True):

		mode = self.atomic_densities_mode		
		printif('-- Map atomic densities on %dx%dx%d grid (mode=%s)'%(self.npts[0],self.npts[1],self.npts[2],mode),self.time)

		# prepare the cuda memory
		if self.cuda:

			# try to import pycuda
			try:
				from pycuda import driver, compiler, gpuarray, tools
				import pycuda.autoinit
			except:
				raise ImportError("Error when importing pyCuda in GridTools")

			# book mem on the gpu
			x_gpu = gpuarray.to_gpu(self.x.astype(np.float32))
			y_gpu = gpuarray.to_gpu(self.y.astype(np.float32))
			z_gpu = gpuarray.to_gpu(self.z.astype(np.float32))
			grid_gpu = gpuarray.zeros(self.npts,np.float32)

		# loop over all the data we want
		for atomtype,vdw_rad in self.local_tqdm(self.atomic_densities.items()):

			
			t0 = time()

			# get the contact atom that of the correct type on both chains
			if only_contact:
				index = self.sqldb.get_contact_atoms()
				xyzA = np.array(self.sqldb.get('x,y,z',rowID=index[0],name=atomtype))
				xyzB = np.array(self.sqldb.get('x,y,z',rowID=index[1],name=atomtype))
				
			else:
				# get the atom that are of the correct type on both chains
				xyzA = np.array(self.sqldb.get('x,y,z',chainID='A',name=atomtype))
				xyzB = np.array(self.sqldb.get('x,y,z',chainID='B',name=atomtype))

			tprocess = time()-t0

			t0 = time()
			# if we use CUDA
			if self.cuda:

				# reset the grid
				grid_gpu *= 0

				# get the atomic densities of chain A
				for pos in xyzA:
					x0,y0,z0 = pos.astype(np.float32)
					vdw = np.float32(vdw_rad)
					self.cuda_atomic(vdw,x0,y0,z0,x_gpu,y_gpu,z_gpu,grid_gpu,block=tuple(self.gpu_block),grid=tuple(self.gpu_grid))
					atdensA = grid_gpu.get()

				# reset the grid
				grid_gpu *= 0

				# get the atomic densities of chain B
				for pos in xyzB:
					x0,y0,z0 = pos.astype(np.float32)
					vdw = np.float32(vdw_rad)
					self.cuda_atomic(vdw,x0,y0,z0,x_gpu,y_gpu,z_gpu,grid_gpu,block=tuple(self.gpu_block),grid=tuple(self.gpu_grid))
					atdensB = grid_gpu.get()

			# if we don't use CUDA
			else:

				# init the grid
				atdensA = np.zeros(self.npts)
				atdensB = np.zeros(self.npts)

				# run on the atoms
				for pos in xyzA:
					atdensA += self.densgrid(pos,vdw_rad)

				# run on the atoms
				for pos in xyzB:
					atdensB += self.densgrid(pos,vdw_rad)

			# create the final grid : A - B
			if mode=='diff':
				self.atdens[atomtype] = atdensA-atdensB

			# create the final grid : A + B
			elif mode=='sum':
				self.atdens[atomtype] = atdensA+atdensB

			# create the final grid : A and B
			elif mode=='ind':
				self.atdens[atomtype+'_chainA'] = atdensA
				self.atdens[atomtype+'_chainB'] = atdensB
			else:
				print('Error: Atomic density mode %s not recognized' %mode)
				sys.exit()

			tgrid = time()-t0 
			printif('     Process time %f ms' %(tprocess*1000),self.time)
			printif('     Grid    time %f ms' %(tgrid*1000),self.time)

	# compute the atomic denisties on the grid
	def densgrid(self,center,vdw_radius):

		'''
		the formula is equation (1) of the Koes paper
		Protein-Ligand Scoring with Convolutional NN Arxiv:1612.02751v1
		'''

		x0,y0,z0 = center
		dd = np.sqrt( (self.xgrid-x0)**2 + (self.ygrid-y0)**2 + (self.zgrid-z0)**2 )
		dgrid = np.zeros(self.npts)
		dgrid[dd<vdw_radius] = np.exp(-2*dd[dd<vdw_radius]**2/vdw_radius**2)
		dgrid[ (dd >=vdw_radius) & (dd<1.5*vdw_radius)] = 4./np.e**2/vdw_radius**2*dd[ (dd >=vdw_radius) & (dd<1.5*vdw_radius)]**2 - 12./np.e**2/vdw_radius*dd[ (dd >=vdw_radius) & (dd<1.5*vdw_radius)] + 9./np.e**2
		return dgrid

	################################################################
	# Residue or Atomic features
	# read the file provided in input 
	# and map it on the grid
	################################################################

	# map residue a feature on the grid
	def map_features(self, featlist, transform=None):

		'''
		For residue based feature the feature file must be of the format 
		chainID    residue_name(3-letter)     residue_number     [values] 

		For atom based feature it must be
		chainID    residue_name(3-letter)     residue_number   atome_name  [values]
		'''

		# declare the total dictionary
		dict_data = {}

		# prepare the cuda memory
		if self.cuda:

			# try to import pycuda
			try:
				from pycuda import driver, compiler, gpuarray, tools
				import pycuda.autoinit
			except:
				raise ImportError("Error when importing pyCuda in GridTools")

			# book mem on the gpu
			x_gpu = gpuarray.to_gpu(self.x.astype(np.float32))
			y_gpu = gpuarray.to_gpu(self.y.astype(np.float32))
			z_gpu = gpuarray.to_gpu(self.z.astype(np.float32))
			grid_gpu = gpuarray.zeros(self.npts,np.float32)

		# loop over all the features required
		for feature_name in featlist:

			
			printif('-- Map %s on %dx%dx%d grid ' %(feature_name,self.npts[0],self.npts[1],self.npts[2]),self.time)

			# read the data
			featgrp = self.molgrp['features']
			if feature_name in featgrp.keys():
				data = featgrp[feature_name].value
			else:
				print('Error Feature not found \n\tPossible features : ' + ' | '.join(featgrp.keys()) )
				raise ValueError('feature %s  not found in the file' %(feature_name))
			

			# detect if we have a xyz format
			# or a byte format
			# define how many elements (ntext) 
			# are present before the feature values
			# xyz : 4 (chain x y z)
			# byte - residue : 3 (chain resSeq resName)
			# byte - atomic  : 4 (chain resSeq resName name)
			if not isinstance(data[0],bytes):
				feature_type = 'xyz'
				ntext = 4
			else:
				try :
					float(data[0].split()[3])
					feature_type = 'residue'
					ntext = 3
				except:
					feature_type = 'atomic' 
					ntext = 4

			# test if the transform is callable
			# and test it on the first line of the data
			# get the data on the first line
			if feature_type != 'xyz':
				data_test = data[0].split()
				data_test = list(map(float,data_test[ntext:]))
			else:
				data_test = data[0,ntext:]

			# define the length of the output
			if transform == None:
				nFeat = len(data_test)
			elif callable(transform):
				nFeat = len(transform(data_test))
			else:
				print('Error transform in map_feature must be callable')
				return None			

			# declare the dict
			# that will in fine holds all the data
			if nFeat == 1:
				if self.feature_mode == 'ind':
					dict_data[feature_name+'_chainA'] = np.zeros(self.npts)
					dict_data[feature_name+'_chainB'] = np.zeros(self.npts)
				else:
					dict_data[feature_name] = np.zeros(self.npts)
			else:
				for iF in range(nFeat):
					if self.feature_mode == 'ind':
						dict_data[feature_name+'_chainA_%03d' %iF] = np.zeros(self.npts)
						dict_data[feature_name+'_chainB_%03d' %iF] = np.zeros(self.npts)
					else:
						dict_data[feature_name+'_%03d' %iF] = np.zeros(self.npts)

			# rest the grid and get the x y z values
			if self.cuda:
				grid_gpu *= 0

			# timing
			tprocess = 0
			tgrid = 0

			# map all the features
			for line in self.local_tqdm(data):
				t0 = time()
				# if the feature was written with xyz data
				# i.e chain x y z values
				if feature_type == 'xyz':

					chain = ['A','B'][int(line[0])]
					pos = line[1:ntext]
					feat_values = np.array(line[ntext:])

				# if the feature was written with bytes
				# i.e chain resSeq resName (name) values
				else:

					# decode the line
					line = line.decode('utf-8').split()

					# get the position of the resnumber
					chain,resName,resNum = line[0],line[1],line[2]

					# get the atom name for atomic data
					if feature_type == 'atomic':
						atName = line[3]

					# get the position
					if feature_type == 'residue':
						pos = np.mean(np.array(self.sqldb.get('x,y,z',chainID=chain,resSeq=resNum)),0)
						sql_resName = list(set(self.sqldb.get('resName',chainID=chain,resSeq=resNum)))
					else:
						pos = np.array(self.sqldb.get('x,y,z',chainID=chain,resSeq=resNum,name=atName))[0]
						sql_resName = list(set(self.sqldb.get('resName',chainID=chain,resSeq=resNum,name=atName)))

					# check if  the resname correspond
					if len(sql_resName) == 0:
						print('Error : SQL query returned empty list')
						print('Tip   : Make sure the parameter file %s' %(feature_file))
						print('Tip   : corresponds to the pdb file %s' %(self.sqldb.pdbfile))
						sys.exit()
					else:
						sql_resName = sql_resName[0]

					if resName != sql_resName:
						print('Residue Name Error in the Feature file %s' %(feature_file))
						print('Feature File : chain %s resNum %s  resName %s' %(chain,resNum, resName))
						print('SQL data     : chain %s resNum %s  resName %s' %(chain,resNum, sql_resName))
						sys.exit()

					# get the values of the feature(s) for thsi residue
					feat_values = np.array(list(map(float,line[ntext:])))

				# postporcess the data
				if callable(transform):
					feat_values = transform(feat_values)

				# handle the mode
				fname = feature_name
				if self.feature_mode == "diff":
					coeff = {'A':1,'B':-1}[chain]
				else:
					coeff = 1
				if self.feature_mode == "ind":
					fname = feature_name + "_chain" + chain
				tprocess += time()-t0

				t0 = time()
				# map this feature(s) on the grid(s)
				if not self.cuda:
					if nFeat == 1:
						dict_data[fname] += coeff*self.featgrid(pos,feat_values)
					else:
						for iF in range(nFeat):
							dict_data[fname+'_%03d' %iF] += coeff*self.featgrid(pos,feat_values[iF])

				# try to use cuda to speed it up		
				else:
					if nFeat == 1:
						x0,y0,z0 = pos.astype(np.float32)
						alpha = np.float32(coeff*feat_values)
						self.cuda_func(alpha,x0,y0,z0,x_gpu,y_gpu,z_gpu,grid_gpu,block=tuple(self.gpu_block),grid=tuple(self.gpu_grid))
					else:
						raise ValueError('CUDA only possible for single-valued features so far')

				tgrid += time()-t0

			if self.cuda:
				dict_data[fname] = grid_gpu.get()
				driver.Context.synchronize()

			
			printif('     Process time %f ms' %(tprocess*1000),self.time)
			printif('     Grid    time %f ms' %(tgrid*1000),self.time)

		return dict_data

	# compute the a given feature on the grid
	def featgrid(self,center,value,type_='fast_gaussian'):

		'''
		map a given feature (atomic or residue) on the grid
		center is the center  of the fragment (pos of the atom or center of the resiude)
		value is the value of the feature
		'''

		# shortcut for th center
		x0,y0,z0 = center

		# simple Gaussian
		if type_ == 'gaussian':
			beta = 1.0/np.max(self.res)
			dd = np.sqrt( (self.xgrid-x0)**2 + (self.ygrid-y0)**2 + (self.zgrid-z0)**2 )
			dd = value*np.exp(-beta*dd)
			return dd

		# fast gaussian
		elif type_ == 'fast_gaussian':

			beta = 1.0/np.max(self.res)
			cutoff = 5.*beta

			dd = np.sqrt( (self.xgrid-x0)**2 + (self.ygrid-y0)**2 + (self.zgrid-z0)**2 )
			dgrid = np.zeros(self.npts)

			dgrid[dd<cutoff] = value*np.exp(-beta*dd[dd<cutoff])

			return dgrid

		# Bsline
		elif type_ == 'bspline':
			spline_order=4
			spl = bspline( (self.xgrid-x0)/self.res[0],spline_order ) * bspline( (self.ygrid-y0)/self.res[1],spline_order ) * bspline( (self.zgrid-z0)/self.res[2],spline_order )
			dd = value*spl
			return dd

		# nearest neighbours
		elif type_ == 'nearest': 

			# distances
			dx,dy,dz = np.abs(self.x-x0),np.abs(self.y-y0),np.abs(self.z-z0)

			# index
			indx = np.argsort(dx)[:2]
			indy = np.argsort(dy)[:2]
			indz = np.argsort(dz)[:2]

			# weight
			wx = dx[indx]
			wx /= np.sum(wx)

			wy = dy[indy]
			wy /= np.sum(wy)

			wz = dx[indz]
			wz /= np.sum(wz)

			# define the points
			indexes = [indx,indy,indz]
			points = list(itertools.product(*indexes))

			# define the weight
			W = [wx,wy,wz]
			W = list(itertools.product(*W))
			W = [np.sum(iw) for iw in W]

			# put that on the grid
			dgrid = np.zeros(self.npts)

			for w,pt in zip(W,points):
				dgrid[pt[0],pt[1],pt[2]] = w*value

			return dgrid

		# default
		else:
			raise ValueError('Options not recognized for the grid',type_)


	################################################################
	# export the grid points for external calculations of some
	# features. For example the electrostatic potential etc ...
	################################################################

	def export_grid_points(self):

		# or to the the hdf5
		grd = self.hdf5.require_group(self.mol_basename+'/grid_points')
		grd.create_dataset('x',data=self.x)
		grd.create_dataset('y',data=self.y)
		grd.create_dataset('z',data=self.z)


	# save the data in the hdf5 file
	def hdf5_grid_data(self,dict_data,data_name):

		# get the group og the feature
		feat_group = self.hdf5.require_group(self.mol_basename+'/mapped_features/'+data_name)

		# gothrough all the feature elements
		for key,value in dict_data.items():

			# remove only subgroup
			if key in feat_group:
				del feat_group[key]

			# create new one
			sub_feat_group = feat_group.create_group(key)

			# try  a sparse representation
			if self.try_sparse:

				# check if the grid is sparse or not
				t0 = time()
				spg = sparse.FLANgrid()
				spg.from_dense(value,beta=1E-2)
				if self.time:
					print('      SPG time %f ms' %((time()-t0)*1000))

				# if we have a sparse matrix
				if spg.sparse:
					sub_feat_group.attrs['sparse'] = spg.sparse
					sub_feat_group.attrs['type'] = 'sparse_matrix'
					sub_feat_group.create_dataset('index',data=spg.index,compression='gzip',compression_opts=9)
					sub_feat_group.create_dataset('value',data=spg.value,compression='gzip',compression_opts=9)

				else:
					sub_feat_group.attrs['sparse'] = spg.sparse
					sub_feat_group.attrs['type'] = 'sparse_matrix'
					sub_feat_group.create_dataset('value',data=spg.value,compression='gzip',compression_opts=9)				
				
			else:
				sub_feat_group.attrs['sparse'] = False
				sub_feat_group.attrs['type'] = 'sparse_matrix'
				sub_feat_group.create_dataset('value',data=value,compression='gzip',compression_opts=9)				

########################################################################################################



