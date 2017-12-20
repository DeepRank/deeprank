import numpy as np
from deeprank.tools import pdb2sql

class SASA(object):


	'''
	Simple class that computes SASA following some of the methods presented in
	
	Solvent accessible surface area approximations for rapid and accurate protein structure prediction
	J Mol Model (2009) 15:1093–1108
	DOI 10.1007/s00894-009-0454-9
	'''

	def __init__(self,pdbfile):

		self.pdbfile = pdbfile

	def get_center(self,chainA='A',chainB='B',center='cb'):

		if center == 'center':
			self.get_residue_center(chainA=chainA,chainB=chainB)
		elif center == 'cb':
			self.get_residue_carbon_beta(chainA=chainA,chainB=chainB)
		else:
			raise ValueError('Options %s not recognized in SASA.get_center' %center)


	def get_residue_center(self,chainA='A',chainB='B'):

		sql = pdb2sql(self.pdbfile)
		resA = np.array(sql.get('resSeq,resName,x,y,z',chainID=chainA))
		resB = np.array(sql.get('resSeq,resName,x,y,z',chainID=chainB))
		sql.close()

		resSeqA = np.unique(resA[:,0].astype(np.int))
		resSeqB = np.unique(resB[:,0].astype(np.int))

		self.xyz = {}
		self.xyz[chainA] = [ np.mean( resA[np.argwhere(resA[:,0].astype(np.int)==r),2:],0 ).astype(np.float).tolist()[0] for r in resSeqA ]
		self.xyz[chainB] = [ np.mean( resB[np.argwhere(resB[:,0].astype(np.int)==r),2:],0 ).astype(np.float).tolist()[0] for r in resSeqB ]

		self.resinfo = {}
		self.resinfo[chainA] = []
		res_seen = set()
		for r in resA[:,:2]:
			if r not in res_seen:
				seen.add(r)
				self.resinfo[chainA].append(r)

		res_seen = set()
		self.resinfo[chainB] = []
		for r in resB[:,:2]:
			if r not in res_seen:
				seen.add(r)
				self.resinfo[chainB].append(r)

	def get_residue_carbon_beta(self,chainA='A',chainB='B'):

		sql = pdb2sql(self.pdbfile)
		resA = np.array(sql.get('resSeq,resName,x,y,z',name='CB',chainID=chainA))
		resB = np.array(sql.get('resSeq,resName,x,y,z',name='CB',chainID=chainB))
		sql.close()

		assert resA[:,0].astype(np.int).tolist() == np.unique(resA[:,0].astype(np.int)).tolist()
		assert resB[:,0].astype(np.int).tolist() == np.unique(resB[:,0].astype(np.int)).tolist()

		self.xyz = {}
		self.xyz[chainA] = resA[:,2:].astype(np.float)
		self.xyz[chainB] = resB[:,2:].astype(np.float)

		self.resinfo = {}
		self.resinfo[chainA] = resA[:,:2]
		self.resinfo[chainB] = resB[:,:2]

	def neighbor_vector(self,lbound=3.3,ubound=11.1,chainA='A',chainB='B',center='cb'):

		# get the center
		self.get_center(chainA=chainA,chainB=chainB,center=center)

		NV = {}

		for chain in [chainA,chainB]:

			for i,xyz in enumerate(self.xyz[chain]):

				vect = self.xyz[chain]-xyz
				dist = np.sqrt(np.sum((self.xyz[chain]-xyz)**2,1))	

				dist = np.delete(dist,i,0)
				vect = np.delete(vect,i,0)
				
				vect /= np.linalg.norm(vect,axis=1).reshape(-1,1)
				
				weight = self.neighbor_weight(dist,lbound=lbound,ubound=ubound).reshape(-1,1)
				vect *= weight

				vect = np.sum(vect,0)
				vect /= np.sum(weight)

				resSeq,resName = self.resinfo[chain][i].tolist()
				key = tuple([chain,int(resSeq),resName])
				value =  np.linalg.norm(vect) 
				NV[key]  = value

		return NV

		
	def neighbor_count(self,lbound=4.0,ubound=11.4,chainA='A',chainB='B',center='cb'):

		# get the center
		self.get_center(chainA=chainA,chainB=chainB,center=center)

		# dict of NC
		NC = {}

		for chain in [chainA,chainB]:
			
			for i,xyz in enumerate(self.xyz[chain]):
				dist = np.sqrt(np.sum((self.xyz[chain]-xyz)**2,1))	
				resSeq,resname = self.resinfo[chain][i].tolist()
				key = tuple([chain,int(resSeq),resName])
				value =  np.sum(self.neighbor_weight(dist,lbound,ubound))
				NC[key]  = value

		return NC

	@staticmethod
	def neighbor_weight(dist,lbound,ubound):
		ind = np.argwhere(  (dist>lbound) & (dist<ubound) )
		dist[ind] = 0.5*( np.cos( np.pi*(dist[ind]-lbound)/(ubound-lbound) ) + 1 )
		dist[dist<=lbound] = 1
		dist[dist>=ubound] = 0
		return dist


if __name__ == '__main__':

	sasa = SASA('1AK4_1w.pdb')
	NV = sasa.neighbor_vector()
	print(NV)