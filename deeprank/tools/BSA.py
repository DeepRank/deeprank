import freesasa
import os
import numpy as np
from deeprank.tools import pdb2sql
from deeprank.tools import FeatureClass


class BSA(FeatureClass):

	'''
	Class to compute the burried surface area
	


	Freesasa is required

	# get the code and go in its dir
	git clone https://github.com/mittinatten/freesasa.git
	cd freesasa

	# configure with fPIC flag on Ubuntu
	./configure --enable-python-binding CFLAGS=-fPIC

	# make the code possibly sudo that
	make 
	make install

	# If the install of the python bindings fails 
	# because no python (problem with anaconda)
	cd ./bindings/python
	python setup.py install



	'''

	def __init__(self,pdb_data,chainA='A',chainB='B'):

		self.pdb_data = pdb_data
		self.sql = pdb2sql(pdb_data)
		self.chains_label =  [chainA,chainB]
		
		self.feature_data = {}
		self.feature_data_xyz = {}


		freesasa.setVerbosity(freesasa.nowarnings)

	def get_structure(self):

		try:
			self.complex = freesasa.Structure(self.pdb_data)
		except:
			self.complex = freesasa.Structure()
			atomdata = self.sql.get('name,resName,resSeq,chainID,x,y,z')
			for atomName,residueName,residueNumber,chainLabel,x,y,z in atomdata:
				atomName = '{:>2}'.format(atomName[0])
				self.complex.addAtom(atomName,residueName,residueNumber,chainLabel,x,y,z)
		self.result_complex = freesasa.calc(self.complex)
		
		self.chains = {}
		self.result_chains = {}
		for label in self.chains_label:
			self.chains[label] = freesasa.Structure()
			atomdata = self.sql.get('name,resName,resSeq,chainID,x,y,z',chainID=label)
			for atomName,residueName,residueNumber,chainLabel,x,y,z in atomdata:
				atomName = '{:>2}'.format(atomName[0])
				self.chains[label].addAtom(atomName,residueName,residueNumber,chainLabel,x,y,z)
			self.result_chains[label] = freesasa.calc(self.chains[label])		
    
	
	def get_contact_residue_sasa(self):

		bsa_data = {}
		bsa_data_xyz = {}

		res = self.sql.get_contact_residue()
		res = res[0]+res[1]

		for r in res:

			# define the selection string and the bsa for the complex
			select_str = ('res, (resi %d) and (chain %s)' %(r[1],r[0]),) 
			asa_complex = freesasa.selectArea(select_str,self.complex,self.result_complex)['res']

			# define the selection string and the bsa for the isolated
			select_str = ('res, resi %d' %r[1],)
			asa_unbound = freesasa.selectArea(select_str,self.chains[r[0]],self.result_chains[r[0]])['res']

			# define the bsa
			bsa = asa_unbound-asa_complex

			# define the xyz key : (chain,x,y,z) 
			chain = {'A':0,'B':1}[r[0]]
			xyz = np.mean(self.sql.get('x,y,z',resSeq=r[1],chainID=r[0]),0)
			xyzkey = tuple([chain]+xyz.tolist())

			# put the data in dict
			bsa_data[r]           =  [bsa]
			bsa_data_xyz[xyzkey]  =  [bsa]

		# pyt the data in dict
		self.feature_data['bsa'] = bsa_data
		self.feature_data_xyz['bsa'] = bsa_data_xyz



if __name__ == '__main__':

	bsa = BSA('1AK4.pdb')
	bsa.get_structure()
	#bsa.get_contact_residue_sasa()
	bsa.sql.close()

