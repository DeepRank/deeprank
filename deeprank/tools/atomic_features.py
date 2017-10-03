import subprocess as sp 
import os
import numpy as np
import sys

import time

from deeprank.tools import pdb2sql
from deeprank.tools import FeatureClass


class atomicFeature(FeatureClass):

	'''
	Sub class that deals with the 
	electrostatic itneraction between atoms
	'''

	def __init__(self,pdbfile,param_charge=None,param_vdw=None,patch_file=None,
		contact_distance=8.5,root_export = './'):

		'''
		subclass the main feature class
		'''
		super().__init__("Atomic")

		# set a few thongs
		self.pdbfile = pdbfile
		self.sqlfile = '_mol.db'
		self.param_charge = param_charge
		self.param_vdw = param_vdw
		self.patch_file = patch_file
		self.contact_distance = contact_distance

		# dircetory to export
		self.root_export = root_export

		# read the pdb as an sql
		self.sqldb = pdb2sql(self.pdbfile,sqlfile=self.sqlfile)

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
		xyz1 = np.array(self.sqldb.get('x,y,z',chain='A'))
		xyz2 = np.array(self.sqldb.get('x,y,z',chain='B'))

		# rowID of the second chain
		index_b =self.sqldb.get('rowID',chain='B')

		# resName of the chains
		resName1 = np.array(self.sqldb.get('resName',chain='A'))
		resName2 = np.array(self.sqldb.get('resName',chain='B'))

		# declare the contact atoms
		self.contact_atoms_A = []
		self.contact_atoms_B = []

		for i,x0 in enumerate(xyz1):

			# compute the contact atoms
			contacts = np.where(np.sqrt(np.sum((xyz2-x0)**2,1)) < self.contact_distance)[0]

			# if we have contact atoms and resA is not a ligand
			if (len(contacts) > 0) and (resName1[i] in self.valid_resnames):

				# add i to the list 
				# add the index of b if its resname is not a ligand
				self.contact_atoms_A += [i]
				self.contact_atoms_B += [index_b[k] for k in contacts if resName2[k] in self.valid_resnames]

		# create a set of unique indexes
		self.contact_atoms_A = list(set(self.contact_atoms_A))
		self.contact_atoms_B = list(set(self.contact_atoms_B))

		if len(self.contact_atoms_A)==0:
			print('Warning : No contact atoms detected in atomicFeature')

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
			query = "WHERE chainID='%s' AND resSeq=%s" %(chain,resNum)
			atNames = np.array(self.sqldb.get('name',query=query))
			rowID = np.array(self.sqldb.get('rowID',query=query))

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
		'''
		new_type = {
		'PROP' : ['all',['HT1','HT2']],
		'NTER' : ['all',['HT1','HT2','HT3']],
		'CTER' : ['all',['OXT']],
		'CTN'  : ['all',['NT','HT1','HT2']],
		'CYNH' : ['CYS',['1SG']],
		'DISU' : ['CYS',['1SG','2SG']],
		'HISD' : ['HIS',['ND1','CE1','CD2','NE2','HE2']],
		'HISE' : ['HIS',['ND1','CE1','CD2','NE2','HE2','HD1']]
		}

		altResName = resName
		for key,values in new_type.items():

			res, atcond = values
			
			if res == resName or res == 'all':
				if all(x in atNames for x in atcond):
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
	#	ELECTROSTATIC
	#
	#####################################################################################


	def compute_coulomb(self):

		print('-- Compute coulomb energy')
		xyz = np.array(self.sqldb.get('x,y,z'))
		charge = np.array(self.sqldb.get('CHARGE'))
		atinfo = self.sqldb.get('chainID,resName,resSeq,name')

		nat = len(xyz)
		matrix = np.zeros((nat,nat))

		for iat in range(nat):

			# coulomb terms
			r = np.sqrt(np.sum((xyz[iat+1:,:]-xyz[iat,:])**2))
			q1q2 = charge[iat]*charge[iat+1:]
			value = q1q2/r

			# store amd symmtrized these values
			matrix[iat,iat+1:] = value
			matrix[iat,:iat] = matrix[:iat,iat]

			# atinfo
			key = tuple(atinfo[iat])

			# store
			value = np.sum(matrix[iat,:])
			self.feature_data[key] = [value]


	def compute_coulomb_interchain_only(self,contact_only=False):

		print('-- Compute coulomb energy interchain only')

		if contact_only:

			if len(self.contact_atoms_A) == 0:
				self.feature_data['coulomb'] = {}
				self.export_directories['coulomb'] = self.root_export+'/ELEC/'
				return

			xyzA = np.array(self.sqldb.get('x,y,z',index=self.contact_atoms_A))
			xyzB = np.array(self.sqldb.get('x,y,z',index=self.contact_atoms_B))

			chargeA = np.array(self.sqldb.get('CHARGE',index=self.contact_atoms_A))
			chargeB = np.array(self.sqldb.get('CHARGE',index=self.contact_atoms_B))

			atinfoA = self.sqldb.get('chainID,resName,resSeq,name',index=self.contact_atoms_A)
			atinfoB = self.sqldb.get('chainID,resName,resSeq,name',index=self.contact_atoms_B)

		else:

			xyzA = np.array(self.sqldb.get('x,y,z',chain='A'))
			xyzB = np.array(self.sqldb.get('x,y,z',chain='B'))

			chargeA = np.array(self.sqldb.get('CHARGE',chain='A'))
			chargeB = np.array(self.sqldb.get('CHARGE',chain='B'))

			atinfoA = self.sqldb.get('chainID,resName,resSeq,name',chain='A')
			atinfoB = self.sqldb.get('chainID,resName,resSeq,name',chain='B')

		natA,natB = len(xyzA),len(xyzB)
		matrix = np.zeros((natA,natB))

		electro_data  = {}

		for iat in range(natA):

			# coulomb terms
			r = np.sqrt(np.sum((xyzB-xyzA[iat,:])**2))
			q1q2 = chargeA[iat]*chargeB
			value = q1q2/r

			# store amd symmtrized these values
			matrix[iat,:] = value
			
			# atinfo
			key = tuple(atinfoA[iat])

			# store
			value = np.sum(value)
			electro_data[key] = [value]

		for iat in range(natB):


			# atinfo
			key = tuple(atinfoB[iat])

			# store
			value = matrix[:,iat]
			value = np.sum(value)
			electro_data[key] = [value]

		# add the feature to the dictionary of features
		self.feature_data['coulomb'] = electro_data
		self.export_directories['coulomb'] = self.root_export+'/ELEC/'


	#####################################################################################
	#
	#	VAN DER WAALS
	#
	#####################################################################################


	def compute_vdw_interchain_only(self,contact_only=False):


		print('-- Compute vdw energy interchain only')

		if contact_only:

			if len(self.contact_atoms_A) == 0:
				self.feature_data['coulomb'] = {}
				self.export_directories['coulomb'] = self.root_export+'/ELEC/'
				return


			xyzA = np.array(self.sqldb.get('x,y,z',index=self.contact_atoms_A))
			xyzB = np.array(self.sqldb.get('x,y,z',index=self.contact_atoms_B))

			vdwA = np.array(self.sqldb.get('eps,sig',index=self.contact_atoms_A))
			vdwB = np.array(self.sqldb.get('eps,sig',index=self.contact_atoms_B))

			epsA,sigA = vdwA[:,0],vdwA[:,1]
			epsB,sigB = vdwB[:,0],vdwB[:,1]

			atinfoA = self.sqldb.get('chainID,resName,resSeq,name',index=self.contact_atoms_A)
			atinfoB = self.sqldb.get('chainID,resName,resSeq,name',index=self.contact_atoms_B)

		else:

			xyzA = np.array(self.sqldb.get('x,y,z',chain='A'))
			xyzB = np.array(self.sqldb.get('x,y,z',chain='B'))

			vdwA = np.array(self.sqldb.get('eps,sig',chain='A'))
			vdwB = np.array(self.sqldb.get('eps,sig',chain='B'))

			epsA,sigA = vdwA[:,0],vdwA[:,1]
			epsB,sigB = vdwB[:,0],vdwB[:,1]

			atinfoA = self.sqldb.get('chainID,resName,resSeq,name',chain='A')
			atinfoB = self.sqldb.get('chainID,resName,resSeq,name',chain='B')

		natA,natB = len(xyzA),len(xyzB)
		matrix = np.zeros((natA,natB))

		vdw_data  = {}

		for iat in range(natA):

			# coulomb terms
			r = np.sqrt(np.sum((xyzB-xyzA[iat,:])**2))
			sigma = 0.5*(sigA[iat] + sigB)
			eps = np.sqrt(epsA[iat]*epsB)

			# normal LJ potential
			value = 4*eps * (  (sigma/r)**12  - (sigma/r)**6 )

			# store these values
			matrix[iat,:] = value
			
			# atinfo
			key = tuple(atinfoA[iat])

			# store
			value = np.sum(value)
			vdw_data[key] = [value]

		for iat in range(natB):

			# atinfo
			key = tuple(atinfoB[iat])

			# store
			value = matrix[:,iat]
			value = np.sum(value)
			vdw_data[key] = [value]

		# add the feature to the dictionary of features
		self.feature_data['vdwaals'] = vdw_data
		self.export_directories['vdwaals'] = self.root_export+'/VDW/'

	def export_data(self):
		bare_mol_name = self.pdbfile.split('/')[-1][:-4]
		super().export_data(bare_mol_name)


