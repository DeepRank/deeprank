import subprocess as sp 
import os
import numpy as np
import sys

import time

from deeprank.tools import pdb2sql
from deeprank.tools import FeatureClass


class AtomicFeature(FeatureClass):

	'''
	Sub class that deals with the electrostatic interaction 
	and van der waals interactions between atoms


	USAGE

	pdbfile : pdb file of the molecule

	param_charge : force field fiel containing the charges e.g. protein-allhdg5.4_new.top
	               must be of the format:

	               CYM  atom O   type=O      charge=-0.500 end
	               ALA    atom N   type=NH1     charge=-0.570 end
	
	param_vdw : force field file containing the vdw parameters e.g  protein-allhdg5.4_new.param
	            must be of the format

	            NONBonded  CYAA    0.105   3.750       0.013    3.750  
	            NONBonded  CCIS    0.105   3.750       0.013    3.750  


	patch_file : valid patch file for the parameters e.g. patch.top
	             The way we handle the patching is very manual and should be
	             made more automatic

	contact_distance : the maximum distance between 2 contact atoms

	include_entire_residue : if atom X of resiue M is a contact atom, include all the atoms 
	                         of resiude M in the contact_atom list

	root_export : root directory where the feature file will be exported

	
	EXAMPLE

	atfeat = atomicFeature(PDB,
	                       param_charge = FF + 'protein-allhdg5-4_new.top',
	                       param_vdw    = FF + 'protein-allhdg5-4_new.param',
	                       patch_file   = FF + 'patch.top')

	# assign the parameters to the atoms
	atfeat.assign_parameters()

	# only compute the pair interactions here
	atfeat.evaluate_pair_interaction(print_interactions=True)
	
	# export the data
	atfeat.export_data()

	# close the sqldb
	atfeat.sqldb.close() 
	

	'''

	def __init__(self,pdbfile,param_charge=None,param_vdw=None,patch_file=None,
		contact_distance=8.5, root_export = './',individual_directory=False,verbose=False):

		'''
		subclass the main feature class
		'''
		super().__init__("Atomic")

		# set a few things
		self.pdbfile = pdbfile
		self.param_charge = param_charge
		self.param_vdw = param_vdw
		self.patch_file = patch_file
		self.contact_distance = contact_distance
		self.individual_directory = individual_directory
		self.verbose = verbose

		# a few constant
		self.eps0 = 1
		self.c = 332.0636

		# dircetory to export
		self.root_export = root_export

		# read the pdb as an sql
		self.sqldb = pdb2sql(self.pdbfile)

		# read the force field
		self.read_charge_file()
		
		if patch_file != None:
			self.patch = self.read_patch()
		else:
			self.patch = None

		# read the vdw param file
		self.read_vdw_file()

		# get the contact atoms
		self.get_contact_atoms()

	#####################################################################################
	#
	#	READ INPUT FILES
	#
	#####################################################################################

	def read_charge_file(self):

		'''
		Read the .top file given in entry
		Create :

			self.charge : dictionary  {(resname,atname):charge}
			self.valid_resnames : list ['VAL','ALP', .....]
			self.at_name_type_convertor : dictionary {(resname,atname):attype}

		'''

		f = open(self.param_charge)
		data = f.readlines()
		f.close()

		# loop over all the data
		self.charge = {}
		self.at_name_type_convertor = {}
		resnames = []

		# loop over the file
		for l in data:

			# split the line
			words = l.split()

			#get the resname/atname
			res,atname = words[0],words[2]

			# get the charge
			ind = l.find('charge=')
			q = float(l[ind+7:ind+13])

			# get the type
			attype = words[3].split('=')[-1]
			
			# store the charge
			self.charge[(res,atname)] = q

			# put the resname in a list so far
			resnames.append(res)

			# dictionary for conversion name/type
			self.at_name_type_convertor[(res,atname)] = attype

		self.valid_resnames = list(set(resnames))
		

	def read_patch(self):

		'''
		Read the patchfile
		Create

			self.patch_charge : Dicitionary	{(resName,atName) : charge}
			self.patch_type   : Dicitionary	{(resName,atName) : type}

		'''

		f = open(self.patch_file)
		data = f.readlines()
		f.close()

		self.patch_charge,self.patch_type = {},{}

		for l in data:

			# ignore comments
			if l[0] != '#' and l[0] != '!' and len(l.split())>0:

				words = l.split()

				# get the new charge
				ind = l.find('CHARGE=')
				q = float(l[ind+7:ind+13])
				self.patch_charge [(words[0],words[3])] = q

				# get the new type if any
				ind = l.find('TYPE=')
				if ind != -1:
					type_ = l[ind+5:ind+9]
					self.patch_type[(words[0],words[3])] = type_.strip()
				


	def read_vdw_file(self):

		'''
		Read the .param file

		NONBONDED ATNAME 0.10000 3.298765 0.100000 3.089222

		First two numbers are for inter-chain interations
		Last two nmbers are for intra-chain interactions
		We only compute the interchain here

		Create 

			self.vdw : dictionary {attype:[E1,S1]}
		'''

		f = open(self.param_vdw)
		data = f.readlines()
		f.close()

		self.vdw_param = {}

		for line in data:

			# split the atom
			line = line.split()

			# empty line
			if len(line) == 0:
				continue

			# comment
			if line[0][0] == '#':
				continue

			self.vdw_param[line[1]] = list(map(float,line[2:4]))


	# get the contact atoms only select amino acids
	# no ligand accounted for
	def get_contact_atoms(self):

		# position of the chains
		xyz1 = np.array(self.sqldb.get('x,y,z',chainID='A'))
		xyz2 = np.array(self.sqldb.get('x,y,z',chainID='B'))

		# rowID of the second chain
		index_b = self.sqldb.get('rowID',chainID='B')

		# resName of the chains
		resName1 = np.array(self.sqldb.get('resName',chainID='A'))
		resName2 = np.array(self.sqldb.get('resName',chainID='B'))

		# declare the contact atoms
		self.contact_atoms_A = []
		self.contact_atoms_B = []

		#The contact atom pairs only co ntains pairs of atoms that are 
		# in contact
		self.contact_pairs = {}

		for i,x0 in enumerate(xyz1):

			# compute the contact atoms
			contacts = np.where(np.sqrt(np.sum((xyz2-x0)**2,1)) < self.contact_distance)[0]

			# if we have contact atoms and resA is not a ligand
			if (len(contacts) > 0) and (resName1[i] in self.valid_resnames):

				# add i to the list 
				# add the index of b if its resname is not a ligand
				self.contact_atoms_A += [i]
				self.contact_atoms_B += [index_b[k] for k in contacts if resName2[k] in self.valid_resnames]

				# add the contact pairs to the list
				self.contact_pairs[i] = [index_b[k] for k in contacts if resName2[k] in self.valid_resnames]

		# create a set of unique indexes
		self.contact_atoms_A = sorted(set(self.contact_atoms_A))
		self.contact_atoms_B = sorted(set(self.contact_atoms_B))

		# if no atoms were found	
		if len(self.contact_atoms_A)==0:
			print('Warning : No contact atoms detected in atomicFeature')


	# for some feature we might want to include entore residue to the contact atoms
	# we can get the indexes of all these atoms with this function
	def _extend_contact_to_residue(self):

		# extract the data
		dataA = self.sqldb.get('chainId,resName,resSeq',rowID=self.contact_atoms_A)
		dataB = self.sqldb.get('chainId,resName,resSeq',rowID=self.contact_atoms_B)

		# create tuple cause we want to hash through it
		dataA = list(map(lambda x: tuple(x),dataA))
		dataB = list(map(lambda x: tuple(x),dataB))

		# extract uniques
		resA = list(set(dataA))
		resB = list(set(dataB))

		# init the list
		index_contact_A,index_contact_B = [],[]

		# contact of chain A
		for resdata in resA:
			chainID,resName,resSeq = resdata
			index_contact_A += self.sqldb.get('rowID',chainID=chainID,resName=resName,resSeq=resSeq)
		
		# contact of chain B
		for resdata in resB:
			chainID,resName,resSeq = resdata
			index_contact_B += self.sqldb.get('rowID',chainID=chainID,resName=resName,resSeq=resSeq)

		# make sure that we don't have double (maybe optional)
		index_contact_A = sorted(set(index_contact_A))
		index_contact_B = sorted(set(index_contact_B))
		
		return index_contact_A,index_contact_B



	#####################################################################################
	#
	#	Assign parameters 
	#
	#####################################################################################

	def assign_parameters(self):

		'''
		Assign to each atom in the pdb its charge and vdw interchain parameters
		Directly deals with the patch so that we don't loop over the residues
		multiple times
		'''

		# get all the resnumbers
		if self.verbose:
			print('-- Assign force field parameters')

		data = self.sqldb.get('chainID,resSeq,resName')
		natom = len(data)
		data = np.unique(np.array(data),axis=0)
		

		# declare the parameters for future insertion in SQL
		atcharge = np.zeros(natom)
		ateps = np.zeros(natom)
		atsig = np.zeros(natom)

		# check 
		attype = np.zeros(natom,dtype='<U5')
		ataltResName = np.zeros(natom,dtype='<U5')


		# add attribute to the db

		# loop over all the residues
		for chain,resNum,resName in data:
			
			# atom types of the residue
			#query = "WHERE chainID='%s' AND resSeq=%s" %(chain,resNum)
			atNames = np.array(self.sqldb.get('name',chainID=chain,resSeq=resNum))
			rowID = np.array(self.sqldb.get('rowID',chainID=chain,resSeq=resNum))

			# get the alternative resname
			altResName = self._get_altResName(resName,atNames)

			# get the charge of this residue
			atcharge[rowID] = self._get_charge(resName,altResName,atNames)

			# get the vdw parameters
			eps,sigma,type_ = self._get_vdw(resName,altResName,atNames)
			ateps[rowID] += eps
			atsig[rowID] += sigma

			ataltResName[rowID] = altResName
			attype[rowID] = type_


		# put the charge in SQL
		self.sqldb.add_column('CHARGE')
		self.sqldb.update_column('CHARGE',atcharge)

		# put the VDW in SQL
		self.sqldb.add_column('eps')
		self.sqldb.update_column('eps',ateps)

		self.sqldb.add_column('sig')
		self.sqldb.update_column('sig',atsig)

		self.sqldb.add_column('type','TEXT')
		self.sqldb.update_column('type',attype)

		self.sqldb.add_column('altRes','TEXT')
		self.sqldb.update_column('altRes',ataltResName)

	def _get_altResName(self,resName,atNames):

		'''
		Apply the patch data
		This is adopted from preScan.pl
		This is very static and I don't quite like it
		The structure of the dictionary is as following

		{ NEWRESTYPE : [ 'OLDRESTYPE' , [atom types that must be present], [atom types that must NOT be present] }   ]  }
		'''

		new_type = {
		'PROP' : ['all',    ['HT1','HT2'],                    [ ]],
		'NTER' : ['all',    ['HT1','HT2','HT3'],              [ ]], 
		'CTER' : ['all',    ['OXT'],                          [ ]],
		'CTN'  : ['all',    ['NT','HT1','HT2'],               [ ]],
		'CYNH' : ['CYS',    ['1SG'],                          ['2SG']],
		'DISU' : ['CYS',    ['1SG','2SG'],                    [ ]],
		'HISE' : ['HIS',    ['ND1','CE1','CD2','NE2','HE2'],  ['HD1']], 
		'HISD' : ['HIS',    ['ND1','CE1','CD2','NE2','HD1'],  ['HE2']]  
		}

		# this works fine now 

		altResName = resName
		for key,values in new_type.items():

			res, atpres, atabs = values
			
			if res == resName or res == 'all':
				if all(x in atNames for x in atpres) and all(x not in atNames for x in atabs):
					altResName = key

		return altResName

	def _get_vdw(self,resName,altResName,atNames):

		# in case the resname is not valid
		if resName not in self.valid_resnames:
			vdw_eps   = [0.00]*len(atNames)
			vdw_sigma = [0.00]*len(atNames)
			type_ = ['None']*len(atNames)

			return vdw_eps,vdw_sigma,type_

		vdw_eps,vdw_sigma,type_ = [],[],[]

		for at in atNames:

			if (altResName,at) in self.patch_type:
				type_.append(self.patch_type[(altResName,at)])
				vdw_data = self.vdw_param[self.patch_type[(altResName,at)]]
				vdw_eps.append(vdw_data[0])
				vdw_sigma.append(vdw_data[1])

			elif (resName,at) in self.at_name_type_convertor:
				type_.append(self.at_name_type_convertor[(resName,at)])
				vdw_data  = self.vdw_param[self.at_name_type_convertor[(resName,at)]]
				vdw_eps.append(vdw_data[0])
				vdw_sigma.append(vdw_data[1])

			else:
				type_.append('None')
				#print('Warning : atom type %s not found for resType %s or patch type %s' %(at,resName,altResName))
				vdw_eps.append(0.0)
				vdw_sigma.append(0.0)

		return vdw_eps,vdw_sigma,type_

	def _get_charge(self,resName,altResName,atNames):

		# in case the resname is not valid
		if resName not in self.valid_resnames:
			q = [0.0]*len(atNames)
			return q

		# assign the charges
		q = []
		for at in atNames:


			if (altResName,at) in self.patch_charge:
				q.append(self.patch_charge[(altResName,at)])


			elif (resName,at) in self.charge:
				q.append(self.charge[(resName,at)])


			else:
				q.append(0.0)

		return q

	#####################################################################################
	#
	#	Simple charges
	#
	#####################################################################################

	def evaluate_charges(self,extend_contact_to_residue=False):

		if self.verbose:
			print('-- Compute list charge for contact atoms only')

		if len(self.contact_atoms_A) == 0:
			self.feature_data['charge'] = {}
			self.export_directories['charge'] = self.root_export+'/CHARGE/'
			return

		# extract information from the pdb2sq
		xyz = np.array(self.sqldb.get('x,y,z'))
		atinfo = self.sqldb.get('chainID,resName,resSeq,name')

		charge = np.array(self.sqldb.get('CHARGE'))

		# define the dictionaries
		charge_data = {}


		# total energy terms
		charge_tot = 0


		# entire residue or not
		if extend_contact_to_residue:
			indA,indB = self._extend_contact_to_residue()
			index_contact_atoms = indA+indB
		else:
			index_contact_atoms = self.contact_atoms_A + self.contact_atoms_B


		# loop over the chain A
		for i in index_contact_atoms:

			# atinfo
			key = tuple(atinfo[i])

			# store in the dicts
			charge_data[key] = [charge[i]]

		# add the electrosatic feature
		self.feature_data['charge'] = charge_data
		if self.individual_directory:
			self.export_directories['charge'] = self.root_export+'/CHARGE/'
		else:
			self.export_directories['charge'] = self.root_export 

	#####################################################################################
	#
	#	PAIR INTERACTIONS
	#
	#####################################################################################

	def evaluate_pair_interaction(self,print_interactions=False,save_interactions=False):

		if self.verbose:
			print('-- Compute interaction energy for contact pairs only')
		
		if len(self.contact_atoms_A) == 0:
			self.feature_data['coulomb'] = {}
			self.export_directories['coulomb'] = self.root_export+'/ELEC/'
			return

		# extract information from the pdb2sq
		xyz = np.array(self.sqldb.get('x,y,z'))
		atinfo = self.sqldb.get('chainID,resName,resSeq,name')

		charge = np.array(self.sqldb.get('CHARGE'))
		vdw = np.array(self.sqldb.get('eps,sig'))
		eps,sig = vdw[:,0],vdw[:,1]
			
		# define the dictionaries
		electro_data = {}
		vdw_data = {}

		# define the matrices 
		natA,natB = len(self.sqldb.get('x',chainID='A')),len(self.sqldb.get('x',chainID='B'))
		matrix_elec = np.zeros((natA,natB))
		matrix_vdw = np.zeros((natA,natB))

		# handle the export of the interaction breakdown
		_save_ = False
		if save_interactions != False:
			if save_interactions == True:
				save_interactions = './'
			if os.path.isdir(save_interactions):
				fname = save_interactions + '/atomic_pair_interaction.dat'
			else:
				fname = save_interactions
			f = open(fname,'w')
			_save_ = True

		# total energy terms
		ec_tot,evdw_tot = 0,0
		
		# loop over the chain A
		for iA,indsB in self.contact_pairs.items():

			# coulomb terms
			r = np.sqrt(np.sum((xyz[indsB,:]-xyz[iA,:])**2,1))
			q1q2 = charge[iA]*charge[indsB]
			ec = q1q2 * self.c / (self.eps0*r) * (1 - (r/self.contact_distance)**2 ) **2
			
			# coulomb terms
			sigma_avg = 0.5*(sig[iA] + sig[indsB])
			eps_avg = np.sqrt(eps[iA]*eps[indsB])

			# normal LJ potential
			evdw = 4.0 *eps_avg * (  (sigma_avg/r)**12  - (sigma_avg/r)**6 ) * self._prefactor_vdw(r)

			#total energy terms
			ec_tot += np.sum(ec)
			evdw_tot += np.sum(evdw)

			# atinfo
			keyA = tuple(atinfo[iA])

			# store in matrix form so that
			# we don't have to recalculate for B
			indb_matrix = [i - natA for i in indsB]
			matrix_elec[iA,indb_matrix] = ec
			matrix_vdw[iA,indb_matrix]  = evdw

			# store in the dicts
			electro_data[keyA] = [np.sum(ec)]
			vdw_data[keyA] = [np.sum(evdw)]

			# print the result
			if _save_ or print_interactions:

				for iB,indexB in enumerate(indsB):

					line = ''
					keyB = tuple(atinfo[indexB])

					line += '{:<3s}'.format(keyA[0])
					line += '\t{:>1s}'.format(keyA[1])
					line += '\t{:>4d}'.format(keyA[2])
					line += '\t{:^4s}'.format(keyA[3])

					line += '\t{:<3s}'.format(keyB[0])
					line += '\t{:>1s}'.format(keyB[1])
					line += '\t{:>4d}'.format(keyB[2])
					line += '\t{:^4s}'.format(keyB[3])	

					line += '\t{: 6.3f}'.format(r[iB])
					line += '\t{: f}'.format(ec[iB])
					line += '\t{: e}'.format(evdw[iB])	

					# print and/or save the interactions
					if print_interactions:
						print(line)

					if _save_:
						line += '\n'	
						f.write(line)

		# print the total interactions
		if print_interactions or _save_:
			line='\n\n'
			line += 'Total Evdw  = {:> 12.8f}\n'.format(evdw_tot)
			line += 'Total Eelec = {:> 12.8f}\n'.format(ec_tot)
			if print_interactions:
				print(line)
			if _save_:
				f.write(line)

		# close export file
		if _save_:
			f.close()

		# loop over the B atoms
		for indexB in self.contact_atoms_B:

			# atinfo
			keyB = tuple(atinfo[indexB])

			# extract the values from the matrix
			ec = matrix_elec[:,indexB-natA]
			evdw = matrix_vdw[:,indexB-natA]

			# store in the dict
			electro_data[keyB] = [np.sum(ec)]
			vdw_data[keyB] = [np.sum(evdw)]			

		# add the electrosatic feature
		self.feature_data['coulomb'] = electro_data
		if self.individual_directory:
			self.export_directories['coulomb'] = self.root_export+'/ELEC/'
		else:
			self.export_directories['coulomb'] = self.root_export

		# add the vdw feature
		self.feature_data['vdwaals'] = vdw_data
		if self.individual_directory:
			self.export_directories['vdwaals'] = self.root_export+'/VDW/'
		else:
			self.export_directories['vdwaals'] = self.root_export


	#####################################################################################
	#
	#	ELECTROSTATIC
	#
	#####################################################################################

	def compute_coulomb_interchain_only(self,dosum=True,contact_only=False):

		if self.verbose:
			print('-- Compute coulomb energy interchain only')

		if contact_only:

			if len(self.contact_atoms_A) == 0:
				self.feature_data['coulomb'] = {}
				self.export_directories['coulomb'] = self.root_export+'/ELEC/'
				return

			xyzA = np.array(self.sqldb.get('x,y,z',rowID=self.contact_atoms_A))
			xyzB = np.array(self.sqldb.get('x,y,z',index=self.contact_atoms_B))

			chargeA = np.array(self.sqldb.get('CHARGE',rowID=self.contact_atoms_A))
			chargeB = np.array(self.sqldb.get('CHARGE',rowID=self.contact_atoms_B))

			atinfoA = self.sqldb.get('chainID,resName,resSeq,name',rowID=self.contact_atoms_A)
			atinfoB = self.sqldb.get('chainID,resName,resSeq,name',rowID=self.contact_atoms_B)

		else:

			xyzA = np.array(self.sqldb.get('x,y,z',chainID='A'))
			xyzB = np.array(self.sqldb.get('x,y,z',chainID='B'))

			chargeA = np.array(self.sqldb.get('CHARGE',chainID='A'))
			chargeB = np.array(self.sqldb.get('CHARGE',chainID='B'))

			atinfoA = self.sqldb.get('chainID,resName,resSeq,name',chainID='A')
			atinfoB = self.sqldb.get('chainID,resName,resSeq,name',chainID='B')

		natA,natB = len(xyzA),len(xyzB)
		matrix = np.zeros((natA,natB))

		electro_data  = {}

		for iat in range(natA):

			# coulomb terms
			r = np.sqrt(np.sum((xyzB-xyzA[iat,:])**2,1))
			q1q2 = chargeA[iat]*chargeB
			value = q1q2/r

			# store amd symmtrized these values
			matrix[iat,:] = value
			
			# atinfo
			key = tuple(atinfoA[iat])

			# store
			if dosum:
				value = [np.sum(value)]
			electro_data[key] = value

		for iat in range(natB):


			# atinfo
			key = tuple(atinfoB[iat])

			# store
			value = matrix[:,iat]
			if dosum:
				value = [np.sum(value)]
			electro_data[key] = value

		# add the feature to the dictionary of features
		self.feature_data['coulomb'] = electro_data
		if self.individual_directory:
			self.export_directories['coulomb'] = self.root_export+'/ELEC/'
		else:
			self.export_directories['coulomb'] = self.root_export


	#####################################################################################
	#
	#	VAN DER WAALS
	#
	#####################################################################################


	def compute_vdw_interchain_only(self,dosum=True,contact_only=False):

		if self.verbose:
			print('-- Compute vdw energy interchain only')

		if contact_only:

			if len(self.contact_atoms_A) == 0:
				self.feature_data['coulomb'] = {}
				self.export_directories['coulomb'] = self.root_export+'/ELEC/'
				return


			xyzA = np.array(self.sqldb.get('x,y,z',rowID=self.contact_atoms_A))
			xyzB = np.array(self.sqldb.get('x,y,z',rowID=self.contact_atoms_B))

			vdwA = np.array(self.sqldb.get('eps,sig',rowID=self.contact_atoms_A))
			vdwB = np.array(self.sqldb.get('eps,sig',rowID=self.contact_atoms_B))

			epsA,sigA = vdwA[:,0],vdwA[:,1]
			epsB,sigB = vdwB[:,0],vdwB[:,1]

			atinfoA = self.sqldb.get('chainID,resName,resSeq,name',rowID=self.contact_atoms_A)
			atinfoB = self.sqldb.get('chainID,resName,resSeq,name',rowID=self.contact_atoms_B)

		else:

			xyzA = np.array(self.sqldb.get('x,y,z',chainID='A'))
			xyzB = np.array(self.sqldb.get('x,y,z',chainID='B'))

			vdwA = np.array(self.sqldb.get('eps,sig',chainID='A'))
			vdwB = np.array(self.sqldb.get('eps,sig',chainID='B'))

			epsA,sigA = vdwA[:,0],vdwA[:,1]
			epsB,sigB = vdwB[:,0],vdwB[:,1]

			atinfoA = self.sqldb.get('chainID,resName,resSeq,name',chainID='A')
			atinfoB = self.sqldb.get('chainID,resName,resSeq,name',chainID='B')

		natA,natB = len(xyzA),len(xyzB)
		matrix = np.zeros((natA,natB))

		vdw_data  = {}

		for iat in range(natA):

			# coulomb terms
			r = np.sqrt(np.sum((xyzB-xyzA[iat,:])**2,1))
			sigma = 0.5*(sigA[iat] + sigB)
			eps = np.sqrt(epsA[iat]*epsB)

			# normal LJ potential
			value = 4*eps * (  (sigma/r)**12  - (sigma/r)**6 )

			# store these values
			matrix[iat,:] = value
			
			# atinfo
			key = tuple(atinfoA[iat])

			# store
			if dosum:
				value = [np.sum(value)]
			vdw_data[key] = value

		for iat in range(natB):

			# atinfo
			key = tuple(atinfoB[iat])

			# store
			value = matrix[:,iat]
			if dosum:
				value = [np.sum(value)]
			vdw_data[key] = value

		# add the feature to the dictionary of features
		self.feature_data['vdwaals'] = vdw_data
		if self.individual_directory:
			self.export_directories['vdwaals'] = self.root_export+'/VDW/'
		else:
			self.export_directories['vdwaals'] = self.root_export

	def _prefactor_vdw(self,r):
		r_off,r_on = 8.5,6.5
		r2 = r**2
		pref = (r_off**2-r2)**2 * (r_off**2 - r2 - 3*(r_on**2 - r2)) / (r_off**2-r_on**2)**3
		pref[r>r_off] = 0.
		pref[r<r_on]  = 1.0
		return pref

	#####################################################################################
	#
	#	EXPORT THE DATA
	#
	#####################################################################################

	def export_data(self):
		bare_mol_name = self.pdbfile.split('/')[-1][:-4]
		super().export_data(bare_mol_name)

	def export_data_hdf5(self,featgrp):
		super().export_data_hdf5(featgrp)

#####################################################################################
#
#	test the function
#
#####################################################################################

if __name__ == "__main__":

	path = os.path.dirname(os.path.realpath(__file__))
	FF = path + '/features/forcefield/'

	BM4 = '/home/nico/Documents/projects/deeprank/data/HADDOCK/BM4_dimers/'
	pdb = BM4 + 'decoys_pdbFLs/1AK4/water/1AK4_100w.pdb' 

	atfeat = AtomicFeature(pdb,
		                   param_charge = FF + 'protein-allhdg5-4_new.top',
						   param_vdw    = FF + 'protein-allhdg5-4_new.param',
						   patch_file   = FF + 'patch.top',
						   root_export = './',
						   individual_directory=False)

	atfeat.assign_parameters()

	# only compute the pair interactions here
	atfeat.evaluate_pair_interaction(print_interactions=True)

	# compute the charges
	# here we extand the contact atoms to
	# entire residue containing at least 1 contact atom
	atfeat.evaluate_charges(extend_contact_to_residue=True)

	
	atfeat.sqldb.close()
	