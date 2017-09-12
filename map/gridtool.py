import numpy as np
import subprocess as sp
import os, sys 
import itertools
from scipy.signal import bspline

try:
	from tqdm import tqdm
except:
	def tqdm(x):
		return x 

# the main gridtool class
class GridTools(object):

	'''
	ARGUMENTS

	mol_name : molecule name containing the two proteins docked. MUST BE A PDB FILE
	          
	data_type : 'haddock' or 'zdock' Type of PDB we re tryin to read
	             the corresponding routines are in get_pdb_data.py

	number_of_points : the number of points we want in each direction of the grid

	resolution : the distance (in Angs) between two points we want. 

	atomic_densities : dictionary of atom types cand their vdw radius
					   exemple {'CA':3.5, 'CB':3.0}
	                   The correspondign atomic densities will be mapped on the grid 
	                   and exported

	residue_feature : dictionnary containing the name and the data files of the features
					  exemple : {'PSSM' : [fileA,fileB]}
					  The corresponding features will be mapped on the grid and exorted

	atomic_feature  : Not yet implemented

	export_path : the path where to export the file. 
	              if not specified the files will be exported in the cwd  


	USAGE


	grid = GridTools(mol_name='complex.1.pdb',
		             atomic_densities=['CA'],
		             number_of_points = [30,30,30],
		             resolution = [1.,1.,1.])

	if the export_path already exists and contains the coodinate of the grid 
	the script will compute the features specified on the grid already present 
	in the directory


	OUTPUT : all files are located in export_path

	AtomicDensities.npy : requires export_atomic_densities = True
	                      contains the atomic densities for each atom_type.
						  The format is : Natomtype x Nx x Ny x Nz

	XX_atdens.cube 		: requires export_atomic_densities = True
						  XX is the PDB atom type e.g. CA, CE, .... specified in atomtype_list
						  Cube file containing the atomic densities. 
						  Can be read directly in VMD

	<feature_name>.npy  : if residue_feature or atomic_feature is not NONE
						  contains all the grid data of he corresponding feature
						  The format is : Nfeature x Nx x Ny x Nz
						  for example PSSM.npy contains usually 20 grid_data

	<n>_<feature_name>.cube : Cube file containing the n-th grid data of the
	                          corresponding feature
 
	contact_atoms.xyz   : XYZ file containing the positions of the contact atoms 

	monomer1.pdb        : 
	monomer2.pdb 		: PDB files containing the positions of each monomer
						  Can be used to represent each monomer with a specific color

 	'''

	def __init__(self,mol_name=None,data_type='haddock',
		         number_of_points = [30,30,30], resolution = [1.,1.,1.],
		         atomic_densities=None,residue_feature=None, atomic_feature=None,
		         export_path='./'):
		
		# mol file	
		self.mol = mol_name

		# data type
		self.data_type = data_type

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
		if self.residue_feature is not None:
		

			for feat_name,feat_files in self.residue_feature.items():

				# map the residue features
				dict_data = self.map_residue_features(feat_name,feat_files)

				# save the data
				self.save_grid_data(dict_data,feat_name)

		# if we wnat the atomic densisties
		if self.atomic_densities is not None:

			# compute the atomic densities
			self.map_atomic_densities()

			# export the densities for visuzaliation
			self.save_grid_data(self.atdens,'AtomicDensities')

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
		if self.residue_feature is not None:
		
			for feat_name,feat_files in self.residue_feature.items():

				# map the residue features
				dict_data = self.map_residue_features(feat_name,feat_files)

				# save the data
				self.save_grid_data(dict_data,feat_name)

		# if we want the atomic densisties
		if self.atomic_densities is not None:

			# compute the atomic densities
			self.map_atomic_densities()

			# export the densities for visuzaliation
			self.save_grid_data(self.atdens,'AtomicDensities')				

	################################################################

	def read_pdb(self):

		# extract atom files
		os.system("awk '/ATOM/' %s > _atom.dat" %self.mol)
		f = open('_atom.dat')
		data = f.readlines()
		f.close()
		

		self.atom_xyz,self.atom_type,self.atom_frag,self.atom_resNum = [],[],[],[]
		for l in data:
			self.atom_type.append(l[13:16].split()[0])
			self.atom_resNum.append(float(l[22:26]))
			self.atom_xyz.append([float(l[31:38]),float(l[38:46]),float(l[46:54])])
			self.atom_frag.append(l.split()[-1])
		
		# convert to numpy
		self.atom_type = np.array(self.atom_type)
		self.atom_resNum = np.array(self.atom_resNum)
		self.atom_xyz = np.array(self.atom_xyz)
		self.atom_frag = np.array(self.atom_frag)

		# get the indexes
		self.atom_index = []
		self.atom_index.append(np.where(self.atom_frag=='A')[0])
		self.atom_index.append(np.where(self.atom_frag=='B')[0])

		# create the monomers with split
		cmd = 'split -l %d %s' %(len(self.atom_index[0]),'_atom.dat')
		os.system(cmd)
		os.system('mv xaa %smonomer1.pdb' %(self.export_path))
		os.system('mv xab %smonomer2.pdb' %(self.export_path))

		# clean
		os.system('rm _atom.dat')

	# read the data from the PDB file
	# outdated and replaced by read_pdb
	# kept here just to be sure
	def get_data(self):

		print('-- Read PDB Data File')

		#switch between different data types
		if self.data_type.lower() == 'haddock':
			self.atom_index = get_pdb_data.haddock(self.mol)
		elif self.data_type.lower() == 'zdock':
			self.atom_index = get_pdb_data.zdock(self.mol)
		else:
			print("Format %s not recognized" %self.data_type)
			sys.exit()

		# extract the positions and atom types of the mol
		self.atom_xyz = np.loadtxt('_xyz.dat')
		self.atom_type = np.array([line.rstrip() for line in open('_atomtype.dat')])
		self.atom_resNum= np.loadtxt('_atomResNum.dat')

		# we export the monomers if required
		os.system("awk '/ATOM/' %s > _tmp.dat" %self.mol)
		cmd = 'split -l %d %s' %(len(self.atom_index[0]),'_tmp.dat')
		os.system(cmd)
		os.system('mv xaa %smonomer1.pdb' %(self.export_path))
		os.system('mv xab %smonomer2.pdb' %(self.export_path))

		# remove the temp files
		os.system('rm _tmp.dat _xyz.dat _atomtype.dat _atomResNum.dat')

	# get the contact atoms
	def get_contact_atoms(self):

		xyz1 = self.atom_xyz[self.atom_index[0],:]
		xyz2 = self.atom_xyz[self.atom_index[1],:]
		self.contact_atoms = []
		for i,x0 in enumerate(xyz1):
			contacts = np.where(np.sqrt(np.sum((xyz2-x0)**2,1)) < self.contact_distance)[0]
			if len(contacts) > 0:
				self.contact_atoms += [self.atom_index[0][i]]
				self.contact_atoms += [self.atom_index[1][k] for k in contacts]
		self.contact_atoms = list(set(self.contact_atoms))
		self.center_contact = np.mean(self.atom_xyz[self.contact_atoms,:],0)


	# define the grid points
	# there is an issue maybe with the ordering
	# In order to visualize the data in VMD the Y and X axis must be inverted ... 
	# I keep it like that for now as it should not matter for the CNN
	# and maybe we don't need atomic denisties as features
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
		#self.set_res(self.x[1]-self.x[0],self.y[1]-self.y[0],self.z[1]-self.z[0])


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

		# loop over all the data we want
		for atomtype,vdw_rad in tqdm(self.atomic_densities.items()):

			# get the atom that are of the correct type
			index = np.where(np.array(self.atom_type) == atomtype)[0]

			# init the grid
			self.atdens[atomtype] = np.zeros(self.npts)

			# run on the atoms
			for ind in index:

				if ind in self.atom_index[0]:
					coeff = 1.0
				elif ind in self.atom_index[1]:
					coeff = -1.0

				self.atdens[atomtype] += coeff*self.densgrid(self.atom_xyz[ind,:],vdw_rad)


	# map a residue level feature on the grid
	def map_residue_features(self,feature_name,feature_files):

		'''
		The first file in feature_files should be for chain A and second for chain B
		Since the residue number in the feature_files depend on the chain ID there is a bit
		of trickyness
		'''
		
		print('-- Map %s on %dx%dx%d grid ' %(feature_name,self.npts[0],self.npts[1],self.npts[2]))

		# read all the fefature files and create the data
		# the first item is for protein 1 and the second for protein 2 
		nfeat = 20		
		feat_data = {'A' : {}, 'B': {}}
		for chainID, feature_file in zip(['A','B'],feature_files):

			# read the feature file
			f = open(feature_file)
			tmp = f.readlines()
			f.close()

			# read the data
			for tmp_data in tmp:
				line_data = tmp_data.split()
				if len(line_data)>1:
					if line_data[0].isdigit():
						resID = int(line_data[0])
						feat_data[chainID][resID] = [float(x) for x in line_data[3:3+nfeat]]

		# get the center of each residue
		chain_pos = {'A': self.atom_xyz[self.atom_index[0],:], 'B':self.atom_xyz[self.atom_index[1],:]}
		chain_res = {'A': self.atom_resNum[self.atom_index[0]], 'B': self.atom_resNum[self.atom_index[1]]}
		posRes = {'A':{},'B':{}}

		for chainID,fd in feat_data.items():

			for resID,data in fd.items():
				resIndex = np.where(np.array(chain_res[chainID])==resID)[0]		
				posRes[chainID][resID] = np.mean(chain_pos[chainID][resIndex,:],0)
				
		# map the features
		nres = [len(fd) for chainID,fd in feat_data.items()]
		
		# loop over all the features
		dict_data = {}
		for iF in tqdm(range(nfeat)):

			dict_data['%03d' %iF] = np.zeros(self.npts)

			# loop over all the residue
			for ichain,chainID in enumerate(['A','B']):

				for iR,pos in posRes[chainID].items():

					if ichain == 0:
						coeff = -1
					elif ichain == 1:
						coeff = 1

					dict_data['%03d' %iF] += coeff*self.featgrid(pos,np.max([0,feat_data[chainID][iR][iF]]))
				
		return dict_data


	################################################################

	# export the grid points for external calculations of some
	# features. For example the electrostatic potential etc ...
	def export_grid_points(self):
		fname = self.export_path + '/grid_points'
		np.savez(fname,x=self.x,y=self.y,z=self.z)


	# export the contact atoms for verification
	def export_contact_atoms(self):
		fname = self.export_path + 'contact_atoms.xyz'
		f = open(fname,'w')
		f.write('%d\n\n' %len(self.contact_atoms))
		for i in self.contact_atoms:
			f.write('%d %f %f %f\n' %(6,self.atom_xyz[i,0],self.atom_xyz[i,1],self.atom_xyz[i,2]))
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

if __name__ == "__main__":
	os.system('rm -rf ./test_output/')
	grid = GridTools(mol_name='./test/1CLV_1w.pdb',
		             number_of_points = [30,30,30],
		             resolution = [1.,1.,1.],
		             atomic_densities={'CA':3.5, 'CB':3.5},
		             residue_feature={
		             'PSSM' : ['./test/1CLV.protein1.ResNumPSSM','./test/1CLV.protein2.ResNumPSSM']},
		             export_path = './test/input/')

	