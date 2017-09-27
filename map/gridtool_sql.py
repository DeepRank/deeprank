import numpy as np
import subprocess as sp
import os, sys 
import itertools
from scipy.signal import bspline

from deeprank.tools import pdb2sql

try:
	from tqdm import tqdm
except:
	def tqdm(x):
		return x 

# the main gridtool class
class GridToolsSQL(object):

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

	def __init__(self,mol_name,
		         number_of_points = [30,30,30], resolution = [1.,1.,1.],
		         atomic_densities=None,residue_feature=None, atomic_feature=None,
		         export_path='./'):
		
		# mol file	
		self.mol = mol_name

		# feature files
		self.residue_feature = residue_feature
		self.atomic_feature = atomic_feature

		# find the base name of the molecule
		# remove all the path and the extension
		self.mol_basename = self.mol.split('/')[-1][:-4]

		# export path
		self.export_path = export_path
		if self.export_path != '' and self.export_path[-1] != '/':
			self.export_path += '/'

		# atom we wnat to compute the densities
		self.atomic_densities = atomic_densities

		# parameter of the grid
		self.npts = np.array(number_of_points).astype('int')
		self.res  = np.array(resolution)

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

		# dictionary of the features
		self.residue_features = {}
		self.atomic_features = {}

		# conversion from boh to angs for VMD visualization
		self.bohr2ang = 0.52918

		# contact distance to locate the interface
		self.contact_distance = 4.5

		# if we already have an output containing the grid
		# we update the existing features
		if  os.path.exists(self.export_path+'/grid_points.npz'):
			print('\n= Updating grid data for %s' %(self.mol))
			self.update_feature()

		else:
			print('\n= Creating grid and grid data for %s' %(self.mol))
			if not os.path.isdir(self.export_path):
				os.mkdir(self.export_path)
			self.create_new_data()



	################################################################

	def create_new_data(self):

		# get the position/atom type .. of the complex
		self.read_pdb()

		#get the contact atoms
		self.get_contact_atoms()

		# print the contact atoms
		self.export_contact_atoms()

		# define the grid 
		self.define_grid_points()

		# save the grid points
		self.export_grid_points()

		#map the features
		self.add_all_residue_features()

		# if we wnat the atomic densisties
		self.add_all_atomic_densities()

		# cloe the db file
		self.sqldb.close()

	################################################################

	def update_feature(self):

		# get the position/atom type .. of the complex
		self.read_pdb()

		# read the grid points
		grid = np.load(self.export_path+'/grid_points.npz')
		self.x,self.y,self.z = grid['x'], grid['y'], grid['z']
		self.ygrid,self.xgrid,self.zgrid = np.meshgrid(self.y,self.x,self.z)
		
		# set the resolution/dimension
		self.npts = np.array([len(self.x),len(self.y),len(self.z)])
		self.res = np.array([self.x[1]-self.x[0],self.y[1]-self.y[0],self.z[1]-self.z[0]])

		#map the features
		self.add_all_residue_features()

		# if we want the atomic densisties
		self.add_all_atomic_densities()			

		# cloe the db file
		self.sqldb.close()

	################################################################


	def read_pdb(self):

		self.sqldb = pdb2sql(self.mol,sqlfile='_mol.db')
		self.sqldb.exportpdb(self.export_path + '/monomer1.pdb',chain='A')
		self.sqldb.exportpdb(self.export_path + '/monomer2.pdb',chain='B')


	# get the contact atoms
	def get_contact_atoms(self):

		xyz1 = np.array(self.sqldb.get('x,y,z',chain='A'))
		xyz2 = np.array(self.sqldb.get('x,y,z',chain='B'))

		index_b =self.sqldb.get('rowID',chain='B')

		self.contact_atoms = []
		for i,x0 in enumerate(xyz1):
			contacts = np.where(np.sqrt(np.sum((xyz2-x0)**2,1)) < self.contact_distance)[0]

			if len(contacts) > 0:
				self.contact_atoms += [i]
				self.contact_atoms += [index_b[k] for k in contacts]

		# create a set of unique indexes
		self.contact_atoms = list(set(self.contact_atoms))

		# get the mean xyz position
		self.center_contact = np.mean(np.array(self.sqldb.get('x,y,z',index=self.contact_atoms)),0)



	################################################################

	# add all the residue features to the data
	def add_all_residue_features(self):

		#map the features
		if self.residue_feature is not None:
		
			for feat_name,feat_files in self.residue_feature.items():

				# map the residue features
				dict_data = self.map_residue_features(feat_name,feat_files)

				# save the data
				self.save_grid_data(dict_data,feat_name)

	# add all the atomic densities to the data
	def add_all_atomic_densities(self):

		# if we wnat the atomic densisties
		if self.atomic_densities is not None:

			# compute the atomic densities
			self.map_atomic_densities()

			# export the densities for visuzaliation
			self.save_grid_data(self.atdens,'AtomicDensities')


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
			
	# compute the atomic denisties on the grid
	def densgrid(self,center,vdw_radius):

		'''
		the formula is equation (1) of the Koes paper
		Protein-Ligand Scoring with Convolutional NN Arxiv:1612.02751v1
		'''

		x0,y0,z0 = center
		dd = np.sqrt( (self.xgrid-x0)**2 + (self.ygrid-y0)**2 + (self.zgrid-z0)**2 )
		dd[dd<vdw_radius] = np.exp(-2*dd[dd<vdw_radius]**2/vdw_radius**2)
		dd[ (dd >=vdw_radius) & (dd<1.5*vdw_radius)] = 4./np.e**2/vdw_radius**2*dd[ (dd >=vdw_radius) & (dd<1.5*vdw_radius)]**2 - 12./np.e**2/vdw_radius*dd[ (dd >=vdw_radius) & (dd<1.5*vdw_radius)] + 9./np.e**2
		dd[dd>=vdw_radius] = 0
		return dd

	# compute the a given feature on the grid
	def featgrid(self,center,value,type_='bspline'):

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

		# Bsline
		if type_ == 'bspline':
			spline_order=4
			spl = bspline( (self.xgrid-x0)/self.res[0],spline_order ) * bspline( (self.ygrid-y0)/self.res[1],spline_order ) * bspline( (self.zgrid-z0)/self.res[2],spline_order )
			dd = value*spl
			return dd


	################################################################

	# compute all the atomic densities data
	def map_atomic_densities(self):

		print('-- Map atomic densities on %dx%dx%d grid '%(self.npts[0],self.npts[1],self.npts[2]))

		coeff = {'A':1,'B':-1}

		# loop over all the data we want
		for atomtype,vdw_rad in tqdm(self.atomic_densities.items()):

			# get the atom that are of the correct type
			xyz = np.array(self.sqldb.get('x,y,z',name=atomtype))
			chain = self.sqldb.get('chainID',name=atomtype)

			# init the grid
			self.atdens[atomtype] = np.zeros(self.npts)

			# run on the atoms
			for pos,seq in zip(xyz,chain):

				self.atdens[atomtype] += coeff[seq]*self.densgrid(pos,vdw_rad)




	# map residue a feature on the grid
	def map_residue_features(self,feature_name,feature_file,mode='softplus',chain_coeff={'A':1,'B':-1}):

		'''
		The feature file must be of the format 
		chainID    residue_name(3-letter)     residue_number     [values]
		'''
		print('-- Map %s on %dx%dx%d grid ' %(feature_name,self.npts[0],self.npts[1],self.npts[2]))

		# read the file
		f = open(feature_file)
		data = f.readlines()
		f.close()

		# number of features
		nFeat = len(data[0].split())-3

		# special case of we postprocess the feature
		if mode == 'max':
			nFeat  = 1

		# declare the dict
		dict_data = {}
		for iF in range(nFeat):
			dict_data['%03d' %iF] = np.zeros(self.npts)

		# map all the features
		for line in tqdm(data):

			line = line.split()

			# get the position of the resnumber
			chain,resNum,resName = line[0],line[1],line[2]
			sql_query = "WHERE chainID='{chain}' AND resSeq={resNum}".format(chain=chain,resNum=resNum)
			pos = np.mean(np.array(self.sqldb.get('x,y,z',query=sql_query)),0)

			# check if we the resname correspond
			sql_resName = list(set(self.sqldb.get('resName',query=sql_query)))[0]
			if resName != sql_resName:
				print('Error in the PSSM file %s' %(feature_file))
				print('Residue Name mismatch')
				print('PSSM File : chain %s resNum %s  resName %s' %(chain,resNum, resName))
				print('SQL data  : chain %s resNum %s  resName %s' %(chain,resNum, sql_resName))
				sys.exit()

			# get the values of the feature(s) for thsi residue
			feat_values = np.array(list(map(float,line[3:])))

			# process the values
			if mode=='all':
				continue

			elif mode == 'max':
				feat_values = np.max(feat_values)

			elif mode == 'softplus':
				feat_values = np.log(1+np.exp(feat_values))

			else:
				print('Error feature mode %s not implemented' %mode)
				sys.exit()

			# get the coefficient of the chain
			coeff = chain_coeff[chain]

			# map this feature(s) on the grid(s)
			for iF in range(nFeat):
				dict_data['%03d' %iF] += coeff*self.featgrid(pos,feat_values[iF])

		return dict_data
	
	################################################################

	# export the grid points for external calculations of some
	# features. For example the electrostatic potential etc ...
	def export_grid_points(self):
		fname = self.export_path + '/grid_points'
		np.savez(fname,x=self.x,y=self.y,z=self.z)


	# export the contact atoms for verification
	def export_contact_atoms(self):

		# get the data
		xyz = self.sqldb.get('x,y,z',index=self.contact_atoms)

		# write the data
		fname = self.export_path + 'contact_atoms.xyz'
		f = open(fname,'w')
		f.write('%d\n\n' %len(self.contact_atoms))
		for pos in xyz:
			f.write('%d %f %f %f\n' %(6,pos[0],pos[1],pos[2]))
		f.close()

	# save the data in npz format
	# and VMD or Blender format vor vizualization
	def save_grid_data(self,dict_data,data_name):

		print('-- Export %s data to %s' %(data_name,self.export_path))

		# export a 4D matrix for the 3DConvNet 
		data_array = []
		for key,value in dict_data.items():	
			data_array.append(dict_data[key])
		mol = self.export_path
		fname = mol + '/' + data_name
		np.save(fname,np.array(data_array))

					
########################################################################################################


	