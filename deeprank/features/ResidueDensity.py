import numpy as np
import itertools
from deeprank.tools import pdb2sql
from deeprank.features import FeatureClass


class residue_pair(object):

	def __init__(self,res,rtype):

		self.res = res
		self.type = rtype
		self.density = {'total':0,'polar':0,'apolar':0,'charged':0}
		self.connections = {'polar':[],'apolar':[],'charged':[]}

	def print(self):
		print('')
		print(self.res, ' : ', self.type)
		print('  Residue Density')
		for k,v in self.density.items():
			print('   '+ k + '\t: '+str(v))
		print('  Residue contact')
		for k,keys in self.connections.items():
			if len(keys)>0:
				print('   ' + k + '\t:',end='')
				for i,v in enumerate(keys):
					print(v,end=' ')
					if not (i+1) % 5:
						print('\n\t\t ',end='')
				print('')

class ResidueDensity(FeatureClass):

	def __init__(self,pdb_data,chainA='A',chainB='B'):

		self.pdb_data = pdb_data
		self.sql=pdb2sql(pdb_data)
		self.chains_label = [chainA,chainB]

		self.feature_data = {}
		self.feature_data_xyz = {}

		self.residue_types = {'CYS':'polar','HIS':'polar','ASN':'polar','GLN':'polar','SER':'polar','THR':'polar','TYR':'polar','TRP':'polar',
		                      'ALA':'apolar','PHE':'apolar','GLY':'apolar','ILE':'apolar','VAL':'apolar','MET':'apolar','PRO':'apolar','LEU':'apolar',
		                      'GLU':'charged','ASP':'charged','LYS':'charged','ARG':'charged'}



	def get(self,cutoff=5.5):

		res = self.sql.get_contact_residue(chain1=self.chains_label[0],
			                               chain2=self.chains_label[1],
			                               cutoff = cutoff,
			                               return_contact_pairs=True)

		self.residue_densities = {}

		for key,other_res in res.items():

			# some residues are not amino acids
			if key[2] not in self.residue_types:
				continue

			if key not in self.residue_densities:
				self.residue_densities[key] = residue_pair(key,self.residue_types[key[2]])
			self.residue_densities[key].density['total'] += len(other_res)

			for key2 in other_res:

				# some residues are not amino acids
				if key2[2] not in self.residue_types:
					continue

				self.residue_densities[key].density[self.residue_types[key2[2]]] += 1
				self.residue_densities[key].connections[self.residue_types[key2[2]]].append(key2)

				if key2 not in self.residue_densities:
					self.residue_densities[key2] = residue_pair(key2,self.residue_types[key2[2]])

				self.residue_densities[key2].density['total'] += 1
				self.residue_densities[key2].density[self.residue_types[key[2]]] += 1
				self.residue_densities[key2].connections[self.residue_types[key[2]]].append(key)

	def print(self):

		for key,res in self.residue_densities.items():
			res.print()

	def extract_features(self):

		self.feature_data['RCD_total'] = {}
		self.feature_data_xyz['RCD_total'] = {}

		restype = ['polar','apolar','charged']
		pairtype = [ '-'.join(p) for p in list(itertools.combinations_with_replacement(restype,2))]
		for p in pairtype:
			self.feature_data['RCD_'+p] = {}
			self.feature_data_xyz['RCD_'+p] = {}

		for key,res in self.residue_densities.items():

			# total density in raw format
			self.feature_data['RCD_total'][key] = [res.density['total']]

			# total density in xyz format

			xyz = np.mean(self.sql.get('x,y,z',resSeq=key[1],chainID=key[0]),0).tolist()
			xyz_key = tuple([{'A':0,'B':1}[key[0]]] + xyz)
			self.feature_data_xyz['RCD_total'][xyz_key] = [res.density['total']]

			# iterate through all the connection
			for r in restype:
				pairtype = 'RCD_' + res.type + '-' + r
				if pairtype not in self.feature_data:
					pairtype = 'RCD_' + r + '-' + res.type
				self.feature_data[pairtype][key] = [res.density[r]]
				self.feature_data_xyz[pairtype][xyz_key] = [res.density[r]]

#####################################################################################
#
#	THE MAIN FUNCTION CALLED IN THE INTERNAL FEATURE CALCULATOR
#
#####################################################################################

def __compute_feature__(pdb_data,featgrp,featgrp_raw):

	# create the BSA instance
	resdens = ResidueDensity(pdb_data)

	# get the densities
	resdens.get(cutoff=5.5)

	# extract the features
	resdens.extract_features()

	# export in the hdf5 file
	resdens.export_dataxyz_hdf5(featgrp)
	resdens.export_data_hdf5(featgrp_raw)

#####################################################################################
#
#	IF WE JUST TEST THE CLASS
#
#####################################################################################


if __name__ == '__main__':

	path = '/home/nico/Documents/projects/deeprank/data/HADDOCK/BM4_dimers/decoys_pdbFLs/1EWY/water/'
	rd = ResidueDensity(path+'1EWY_100w.pdb')
	rd.get(cutoff=5.5)
	rd.print()
	rd.extract_features()

