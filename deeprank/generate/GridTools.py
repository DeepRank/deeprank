import numpy as np
import subprocess as sp
import os, sys 
import itertools
from scipy.signal import bspline
from collections import OrderedDict

from deeprank.tools import pdb2sql

try:
	from tqdm import tqdm
except:
	def tqdm(x):
		return x 



# the main gridtool class
class GridTools(object):

	'''
	
	Map the feature of a complex on the grid


	ARGUMENTS

	mol_name

			molecule name containing the two proteins docked. 
			MUST BE A PDB FILE
	          
	number_of_points 

			the number of points we want in each direction of the grid

	resolution

			the distance (in Angs) between two points we want. 

	atomic_densities

			dictionary of atom types cand their vdw radius
			exemple {'CA':3.5, 'CB':3.0}
			The correspondign atomic densities will be mapped on the grid 
			and exported

	residue_feature

			dictionnary containing the name and the data files of the features
			exemple : {'PSSM' : file}
			The corresponding features will be mapped on the grid and exorted

	atomic_feature

			Not yet implemented

	export_path
			
			the path where to export the file. 
			if not specified the files will be exported in the cwd  


	USAGE


	grid = GridTools(mol_name='complex.1.pdb',
		             atomic_densities={'CA':3.5},
		             number_of_points = [30,30,30],
		             resolution = [1.,1.,1.])

	if the export_path already exists and contains the coodinate of the grid 
	the script will compute the features specified on the grid already present 
	in the directory


	OUTPUT : all files are located in export_path

	AtomicDensities.npy

			requires export_atomic_densities = True
			contains the atomic densities for each atom_type.
			The format is : Natomtype x Nx x Ny x Nz


	<feature_name>.npy

			if residue_feature or atomic_feature is not NONE
			contains all the grid data of he corresponding feature
			The format is : Nfeature x Nx x Ny x Nz
			for example PSSM.npy contains usually 20 grid_data
 
	contact_atoms.xyz

			XYZ file containing the positions of the contact atoms 

	monomer1.pdb/momomer2.pdb

			PDB files containing the positions of each monomer
			Can be used to represent each monomer with a specific color
	
	'''

	def __init__(self, molgrp,
				number_of_points = [30,30,30],resolution = [1.,1.,1.],
				atomic_densities=None, atomic_densities_mode='sum',
				residue_feature =None, residue_feature_mode ='sum',
				atomic_feature  =None, atomic_feature_mode  ='sum',
				contact_distance = 8.5, hdf5_file=None,
				cuda=False, gpu_block=None, tune_kernel=False):
		
		# mol file	
		self.molgrp = molgrp

		# feature requestedOO
		self.atomic_densities = atomic_densities
		self.residue_feature = residue_feature
		self.atomic_feature = atomic_feature

		# mapping mode
		self.atomic_feature_mode = atomic_feature_mode
		self.atomic_densities_mode = atomic_densities_mode

		# feature type requested
		self.feattype_required = []
		if self.residue_feature != None:
			self.feattype_required.append('residue')
		if self.atomic_feature != None:
			self.feattype_required.append('atomic')

		# find the base name of the molecule
		# remove all the path and the extension
		if not tune_kernel:
			self.mol_basename = molgrp.name
			
			# hdf5 file
			self.hdf5 = hdf5_file

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
			self.init_cuda(gpu_block)

		# parameter of the atomic system
		self.atom_xyz = None
		self.atom_index = None
		self.atom_type = None

		# grid points
		self.x = None
		self.y = None
		self.z = None

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

		# if we only want to tune the kernel
		if tune_kernel:
			self.tune_kernel()

		# or we do the full thing
		else:

			# if we already have an output containing the grid
			# we update the existing features
			_update_ = False	
			if self.mol_basename+'/grid_points' in self.hdf5:
				_update_ = True

			if _update_:
				print('\n= Updating grid data for %s' %(self.mol_basename))
				self.update_feature()
			else:
				print('\n= Creating grid and grid data for %s' %(self.mol_basename))	
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
		for feattype in self.feattype_required:
			self.add_all_features(feattype)

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
		for feattype in self.feattype_required:
			self.add_all_features(feattype)

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
	def add_all_features(self,feature_type):

		if feature_type == 'residue':
			featlist = self.residue_feature
		elif feature_type == 'atomic':
			featlist = self.atomic_feature
		else:
			print('Error feature type must be residue or atomic')
			return

		#map the features
		if featlist is not None:
		
			# map the residue features
			dict_data = self.map_features(featlist,feature_type)

			# save to hdf5 if specfied
			self.hdf5_grid_data(dict_data,'%sFeature_%s' %(feature_type, self.atomic_feature_mode))

		
	

	# add all the atomic densities to the data
	def add_all_atomic_densities(self):

		# if we wnat the atomic densisties
		if self.atomic_densities is not None:

			# compute the atomic densities
			self.map_atomic_densities()

			# save to hdf5
			self.hdf5_grid_data(self.atdens,'AtomicDensities_%s' %(self.atomic_densities_mode))


			

	################################################################
	# define the grid points
	# there is an issue maybe with the ordering
	# In order to visualize the data in VMD the Y and X axis must be inverted ... 
	# I keep it like that for now as it should not matter for the CNN
	# and maybe we don't need atomic denisties as features
	################################################################

	def define_grid_points(self):

		print('-- Define %dx%dx%d grid ' %(self.npts[0],self.npts[1],self.npts[2]))
		print('-- Resolution of %1.2fx%1.2fx%1.2f Angs' %(self.res[0],self.res[1],self.res[2]))


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
	def map_atomic_densities(self):

		mode = self.atomic_densities_mode
		print('-- Map atomic densities on %dx%dx%d grid (mode=%s)'%(self.npts[0],self.npts[1],self.npts[2],mode))

		# loop over all the data we want
		for atomtype,vdw_rad in tqdm(self.atomic_densities.items()):

			# get the atom that are of the correct type for chain A
			xyzA = np.array(self.sqldb.get('x,y,z',chainID='A',name=atomtype))

			# get the atom that are of the correct type for chain B
			xyzB = np.array(self.sqldb.get('x,y,z',chainID='B',name=atomtype))

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
	def map_features(self, featlist, feature_type, transform=None):

		'''
		For residue based feature the feature file must be of the format 
		chainID    residue_name(3-letter)     residue_number     [values] 

		For atom based feature it must be
		chainID    residue_name(3-letter)     residue_number   atome_name  [values]
		'''

		# declare the total dictionary
		dict_data = {}

		# prepare cuda if we need to
		if self.cuda:

			# book memp on the gpu
			x_gpu = gpuarray.to_gpu(self.x.astype(npfloat32))
			y_gpu = gpuarray.to_gpu(self.y.astype(npfloat32))
			z_gpu = gpuarray.to_gpu(self.z.astype(npfloat32))
			grid_gpu = gpuarray.zeros(self.npts,np.float32)

			# get the kernel
			kernel_code_template = self.get_cuda_kernel()
			kernel_code = kernel_code_template % {'nx' : self.npts[0],  'ny' : self.npts[1], 'nz' : self.npts[2], 'RES' : np.max(self.res) }

			# inpt the block size
			kernel_code = self.prepare_kernel(kernel_code)

			# compile and get the function
			mod = compiler.SourceModule(kernel_code)
			addgrid = mod.get_function('AddGrid')

		# number of features
		if feature_type == 'residue':
			ntext = 3
		elif feature_type == 'atomic':
			ntext = 4
		else:
			print('Error feature type either residue or atomic')
			return None

		# loop over all the features required
		for feature_name in featlist:

			print('-- Map %s on %dx%dx%d grid ' %(feature_name,self.npts[0],self.npts[1],self.npts[2]))

			# read the data
			featgrp = self.molgrp['features']
			if feature_name in featgrp.keys():
				data = featgrp[feature_name].value
			else:
				print('Error Feature not found \n\tPossible features : ' + ' | '.join(featgrp.keys()) )
				raise ValueError('feature %s  not found in the file' %(feature_name))
			
			# get the data on the first line
			data_test = data[0].split()
			data_test = list(map(float,data_test[ntext:]))

			# define the length of the output
			if transform == None:
				nFeat = len(data_test)
			elif callable(transform):
				nFeat = len(transform(data_test))
			else:
				print('Error transform in map_feature must be callable')
				return None			

			# declare the dict
			if nFeat == 1:
				if self.atomic_feature_mode == 'ind':
					dict_data[feature_name+'_chainA'] = np.zeros(self.npts)
					dict_data[feature_name+'_chainB'] = np.zeros(self.npts)
				else:
					dict_data[feature_name] = np.zeros(self.npts)
			else:
				for iF in range(nFeat):
					if self.atomic_feature_mode == 'ind':
						dict_data[feature_name+'_chainA_%03d' %iF] = np.zeros(self.npts)
						dict_data[feature_name+'_chainB_%03d' %iF] = np.zeros(self.npts)
					else:
						dict_data[feature_name+'_%03d' %iF] = np.zeros(self.npts)

			# rest the grid
			if self.cuda:
				grid_gpu *= 0

			# map all the features
			for line in tqdm(data):

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
				if self.atomic_feature_mode == "diff":
					coeff = {'A':1,'B':-1}[chain]
				else:
					coeff = 1
				if self.atomic_feature_mode == "ind":
					fname = feature_name + "_chain" + chain

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
						x0,y0,z0 = pos
						alpha = coeff*feat_values
						addgrid(alpha,x0,y0,z0,x_gpu,y_gpu,z_gpu,grid_gpu,block=tupe(self.gpu_block),grid=tuple(self.gpu_grid))
					else:
						raise ValueError('CUDA only possible for single-valued features so far')

			if self.cuda:
				dict_data[fname] = grid_gpu.get()

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
			cutoff = 10.*beta

			dd = np.sqrt( (self.xgrid-x0)**2 + (self.ygrid-y0)**2 + (self.zgrid-z0)**2 )
			dgrid = np.zeros(self.npts)

			dgrid[dd<cutoff] = np.exp(-beta*dd[dd<cutoff])

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



	##########################################################	
	#	Init the CUDA parameters
	##########################################################
	def init_cuda(self,block):

		from pycuda import driver, compiler, gpuarray, tools
		import pycuda.autoinit
	
		self.gpu_block = block
		if self.gpu_block is not None:
			self.gpu_grid = [ int(np.ceil(n/b)) for b,n in zip(self.gpu_block,self.npts)]

	##########################################################	
	#	Get the CUDA KERNEL
	##########################################################
	@staticmethod
	def get_cuda_kernel():

		return """
		#include <math.h>
		__global__ void AddGrid(float alpha, float x0, float y0, float z0, float *xvect, float *yvect, float *zvect, float *out)
		{
			
			// 3D thread 
		    int tx = threadIdx.x + block_size_x * blockIdx.x;
		    int ty = threadIdx.y + block_size_y * blockIdx.y;
		    int tz = threadIdx.z + block_size_z * blockIdx.z;

		    float beta = 1.0/%(RES)s;

		    if ( ( tx < %(nx)s ) && (ty < %(ny)s) && (tz < %(nz)s) )
		    {

		    	float dx = xvect[tx] - x0;
		    	float dy = yvect[ty] - y0;
		    	float dz = zvect[tz] - z0;
		    	float d = sqrt(dx*dx + dy*dy + dz*dz);
		    	out[ty * %(nx)s * %(nz)s + tx * %(nz)s + tz] += alpha*exp(-beta*d);
		    }
		}
		"""


	#################################################################
	#	Tune the kernel
	#################################################################
	def tune_kernel(self):

		try:
			from kernel_tuner import tune_kernel
		except:
			print('Install the Kernel Tuner : \n \t\t pip install kernel_tuner')
			print('http://benvanwerkhoven.github.io/kernel_tuner/')


		# define the grid
		self.center_contact = np.zeros(3)
		self.define_grid_points()

		# create the dictionary containing the tune parameters
		tune_params = OrderedDict()
		tune_params['block_size_x'] = [2,4,8,16,32]
		tune_params['block_size_y'] = [2,4,8,16,32]
		tune_params['block_size_z'] = [2,4,8,16,32]

		# define the final grid
		grid = np.zeros_like(self.xgrid)

		# arguments of the CUDA function
		x0,y0,z0 = np.float32(0),np.float32(0),np.float32(0)
		args = [x0,y0,z0,self.x,self.y,self.z,grid]

		# dimensionality
		problem_size = self.npts

		# get the kernel
		kernel_code_template = self.get_cuda_kernel()
		kernel_code = kernel_code_template % {'nx' : self.npts[0], 'ny': self.npts[1], 'nz' : self.npts[2], 'RES' : np.max(self.res)}

		# tune
		result = tune_kernel('AddGrid', kernel_code,problem_size,args,tune_params)

	##########################################################	
	#	prepare the kernel
	##########################################################
	def prepare_kernel(kernel_code):

		# change the values of the block sizes in the kernel
		fixed_params = OrderedDict()
		fixed_params['block_size_x'] = self.gpu_block[0]
		fixed_params['block_size_y'] = self.gpu_block[1]
		fixed_params['block_size_z'] = self.gpu_block[2]

		for k,v in fixed_params.items():
		    kernel_code = kernel_code.replace(k,str(v))

		return kernel_code

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

		feat_group = self.hdf5.create_group(self.mol_basename+'/mapped_features/'+data_name)
		for key,value in dict_data.items():
			if key not in feat_group:
				feat_group.create_dataset(key,data=value)
			else:
				tmp = feat_group[key]
				tmp[...] = value

					
########################################################################################################



