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

	def __init__(self,pdbfile,param_charge=None,param_vdw=None,patch_file=None,contact_distance=8.5):

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

			self.patch : Dicitionary	{(resName,atName) : charge}

		'''

		f = open(self.patch_file)
		data = f.readlines()
		f.close()

		self.patch = {}
		for l in data:

			# ignore comments
			if l[0] != '#' and l[0] != '!' and len(l.split())>0:
				words = l.split()
				ind = l.find('CHARGE=')
				q = float(l[ind+7:ind+13])
				self.patch [(words[0],words[3])] = q
		


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


	# get the contact atoms
	def get_contact_atoms(self):

		xyz1 = np.array(self.sqldb.get('x,y,z',chain='A'))
		xyz2 = np.array(self.sqldb.get('x,y,z',chain='B'))

		index_b =self.sqldb.get('rowID',chain='B')

		self.contact_atoms_A = []
		self.contact_atoms_B = []
		for i,x0 in enumerate(xyz1):
			contacts = np.where(np.sqrt(np.sum((xyz2-x0)**2,1)) < self.contact_distance)[0]

			if len(contacts) > 0:
				self.contact_atoms_A += [i]
				self.contact_atoms_B += [index_b[k] for k in contacts]

		# create a set of unique indexes
		self.contact_atoms_A = list(set(self.contact_atoms_A))
		self.contact_atoms_B = list(set(self.contact_atoms_B))


	#####################################################################################
	#
	#	ELECTROSTATIC
	#
	#####################################################################################

	def assign_charges(self):

		'''
		Assign to each atom in the pdb its charge
		'''

		# get all the resnumbers
		print('-- Assign atomic charges')
		data = self.sqldb.get('chainID,resSeq,resName')
		data = np.unique(np.array(data),axis=0)

		# declare the charges
		atcharge = []

		# loop over all the residues
		for chain,resNum,resName in data:
			
			# atom types of the residue
			query = "WHERE chainID='%s' AND resSeq=%s" %(chain,resNum)
			atNames = self.sqldb.get('name',query=query)

			# get the charge of this residue
			atcharge += self._get_charge(resName,atNames)

		self.sqldb.add_column('CHARGE')
		self.sqldb.update_column('CHARGE',atcharge)


	def _get_charge(self,resName,atNames):

		# in case the resname is not valid
		if resName not in self.valid_resnames:
			q = [0]*len(atNames)
			return q

		# assign first the normal charge
		q = []
		for at in atNames:
			if (resName,at) in self.charge:
				q.append(self.charge[(resName,at)])
			else:
				q.append(0)

		# apply the patch for NTER
		altResName = None
		if 'HT1' in atNames and 'HT2' in atNames and 'HT3' in atNames:
			altResName = 'NTER'

		# apply the patch for PROP
		elif 'HT1' in atNames and 'HT2' in atNames:
			altResName = 'PROP'


		# apply the patch for CTER
		elif 'OXT' in atNames:
			altResName = 'CTER'

		# patch for CTN
		elif 'H1' in atNames and 'H2' in atNames and 'NT' in atNames:
			altResName='CTN'

		# apply the patch info	
		if altResName != None and self.patch != None:
			for iat, at in enumerate(atNames):
				if (altResName,at) in self.patch:
					q[iat] = self.patch[(altResName,at)]

		return q


	def compute_coulomb(self):

		print('-- compute coulomb energy')
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

		print('-- compute coulomb energy interchain only')

		if contact_only:

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

	#####################################################################################
	#
	#	VAN DER WAALS
	#
	#####################################################################################

	def assign_vdw(self):

		print('-- Assign vdw paramerers')
		unique_name = np.unique(np.array(self.sqldb.get('resName,name')),axis=0)

		self.sqldb.add_column('eps')
		self.sqldb.add_column('sig')

		for resName,name in unique_name:

			if (resName,name) in self.at_name_type_convertor:
				vdw = self.vdw_param[self.at_name_type_convertor[(resName,name)]]
				query = "WHERE name='{name}' AND resName='{resName}'".format(name=name,resName=resName)
				self.sqldb.put('eps',vdw[0],query=query)
				self.sqldb.put('sig',vdw[1],query=query)


	def compute_vdw_interchain_only(self,contact_only=False):


		print('-- compute vdw energy interchain only')

		if contact_only:

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

	def export_data(self):
		super().export_data()


if __name__ == '__main__':
	
	t0 = time.time()
	atfeat = atomicFeature('./3CPH_129w.pdb',param_charge='protein-allhdg5-4_new.top',
											 param_vdw='protein-allhdg5-4_new.param',
											 patch_file='patch.top')
	atfeat.assign_charges()
	atfeat.compute_coulomb_interchain_only(contact_only=True)
	atfeat.assign_vdw()
	atfeat.compute_vdw_interchain_only(contact_only=True)
	atfeat.export_data()
	print('Done in %f s' %(time.time()-t0))