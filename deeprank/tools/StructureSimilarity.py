import numpy as np
from deeprank.tools import transform, pdb2sql 
import sys

'''
Class to compute the
	L-RMS
	I-RMS
	Fnat
	DockQ score
of a decoy versus a ref
'''

class StructureSimilarity(object):

	def __init__(self,decoy,ref):
		self.decoy = decoy
		self.ref = ref


	# compute the L-RMSD
	def compute_lrmsd(self,exportpath=None):

		'''
		Ref : DockQ: A Quality Measure for Protein-Protein Docking Models
		      http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0161879
		L-RMSD is computed by aligning the longest chain of the decoy to the one of the  reference
		and computing the RMSD of the shortest chain between decoy and ref
		'''

		# create teh sql
		sql_decoy = pdb2sql(self.decoy,sqlfile='decoy.db')
		sql_ref = pdb2sql(self.ref,sqlfile='ref.db')

		# extract the pos of chains A
		xyz_decoy_A = np.array(sql_decoy.get('x,y,z',chain='A'))
		xyz_ref_A = np.array(sql_ref.get('x,y,z',chain='A'))

		# extract the pos of chains B
		xyz_decoy_B = np.array(sql_decoy.get('x,y,z',chain='B'))
		xyz_ref_B = np.array(sql_ref.get('x,y,z',chain='B'))


		# check the lengthes
		if len(xyz_decoy_A) != len(xyz_ref_A):
			#print('Clean the xyz of chain A')
			xyz_decoy_A, xyz_ref_A = self.get_identical_atoms(sql_decoy,sql_ref,'A')

		if len(xyz_decoy_B) != len(xyz_ref_B):
			#print('Clean the xyz of chain B')
			xyz_decoy_B, xyz_ref_B = self.get_identical_atoms(sql_decoy,sql_ref,'B')
		

		# detect which chain is the longest
		nA,nB = len(xyz_decoy_A),len(xyz_decoy_B)
		if nA>nB:
			xyz_decoy_long = xyz_decoy_A
			xyz_ref_long = xyz_ref_A
			
			xyz_decoy_short = xyz_decoy_B
			xyz_ref_short = xyz_ref_B

		else:
			xyz_decoy_long = xyz_decoy_B
			xyz_ref_long = xyz_ref_B
			
			xyz_decoy_short = xyz_decoy_A
			xyz_ref_short = xyz_ref_A

		# get the translation so that both A chains are centered
		tr_decoy = self.get_trans_vect(xyz_decoy_long)
		tr_ref = self.get_trans_vect(xyz_ref_long)

		# translate everything for 1
		xyz_decoy_short = transform.translation(xyz_decoy_short,tr_decoy)
		xyz_decoy_long = transform.translation(xyz_decoy_long,tr_decoy)

		# translate everuthing for 2
		xyz_ref_short = transform.translation(xyz_ref_short,tr_ref)
		xyz_ref_long = transform.translation(xyz_ref_long,tr_ref)

		# get the ideql rotation matrix
		# to superimpose the A chains
		U = self.get_rotation_matrix_Kabsh(xyz_decoy_long,xyz_ref_long)

		# rotate the entire fragment
		xyz_decoy_short = transform.rotation_matrix(xyz_decoy_short,U,center=False)


		# compute the RMSD
		lrmsd =  self.get_rmsd(xyz_decoy_short,xyz_ref_short)

		# export the pdb for verifiactions
		if exportpath is not None:

			# extract the pos of the dimer
			xyz_decoy = np.array(sql_decoy.get('x,y,z'))
			xyz_ref = np.array(sql_ref.get('x,y,z'))

			# translate
			xyz_ref = transform.translation(xyz_ref,tr_ref)
			xyz_decoy = transform.translation(xyz_decoy,tr_decoy)
			
			# rotate decoy
			xyz_decoy= transform.rotation_matrix(xyz_decoy,U,center=False)

			# update the sql database
			sql_decoy.update_xyz(xyz_decoy)
			sql_ref.update_xyz(xyz_ref)		

			# export
			sql_decoy.exportpdb(exportpath+'/lrmsd_decoy.pdb')
			sql_ref.exportpdb(exportpath+'/lrmsd_aligned.pdb')

		# close the db
		sql_decoy.close()
		sql_ref.close()

		return lrmsd

	# compute the irmsd
	def compute_irmsd(self,exportpath=None):

		'''
		Ref : DockQ: A Quality Measure for Protein-Protein Docking Models
		      http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0161879
		i-RMSD is computed selecting the back bone of the contact residue with a cutoff of 10A
		in the decoy. Align these as best as possible with their coutner part in the ref
		and and compute the RMSD
		'''

		# create thes sql
		sql_decoy = pdb2sql(self.decoy,sqlfile='mol1.db')
		sql_ref = pdb2sql(self.ref,sqlfile='mol2.db')

		# get the contact atoms
		cutoff = 10.
		contact_decoy = sql_decoy.get_contact_atoms(cutoff=cutoff,extend_to_residue=True,only_backbone_atoms=True)

		# make a single list
		index_contact_decoy = contact_decoy[0]+contact_decoy[1]


		# get the xyz and atom identifier of the decoy contact atoms
		xyz_contact_decoy = sql_decoy.get('x,y,z',index=index_contact_decoy)
		data_contact_decoy = sql_decoy.get('chainID,resSeq,resName,name',index=index_contact_decoy)

		# get the xyz and atom indeitifier of the reference
		xyz_ref = sql_ref.get('x,y,z')
		data_ref = sql_ref.get('chainID,resSeq,resName,name')

		# loop through the decoy label
		# check if the atom is in the ref
		# if yes -> add xyz to xyz_contact_ref
		# if no  -> remove the corresponding to xyz_contact_decoy
		xyz_contact_ref = []
		index_contact_ref = []
		clean_decoy = False
		for iat,atom in enumerate(data_contact_decoy):

			try:
				index = data_ref.index(atom)
				index_contact_ref.append(index)
				xyz_contact_ref.append(xyz_ref[index])
			except:
				#print('Warning CHAIN %s RES %d RESNAME %s ATOM %s not found in reference' %(atom[0],atom[1],atom[2],atom[3]))
				xyz_contact_decoy[iat] = None
				index_contact_decoy[iat] = None
				clean_decoy = True

		# clean the xyz
		if clean_decoy:
			xyz_contact_decoy = [xyz for xyz in xyz_contact_decoy if xyz is not None]
			index_contact_decoy = [ind for ind in index_contact_decoy if ind is not None]

		# check that we still have atoms in both chains
		chain_decoy = list(set(sql_decoy.get('chainID',index=index_contact_decoy)))
		chain_ref = list(set(sql_ref.get('chainID',index=index_contact_ref)))
		error = 1
		if len(chain_decoy)<2:
			print('Error in i-rmsd: only one chain represented in contact atoms of the decoy')
			error = -1
		if len(chain_ref)<2:
			print('Error in i-rmsd: only one chain represented in contact atoms of the ref')
			error = -1

		# get the translation so that both A chains are centered
		tr_decoy = self.get_trans_vect(xyz_contact_decoy)
		tr_ref = self.get_trans_vect(xyz_contact_ref)

		# translate everything 
		xyz_contact_decoy = transform.translation(xyz_contact_decoy,tr_decoy)
		xyz_contact_ref   = transform.translation(xyz_contact_ref,tr_ref)

		# get the ideql rotation matrix
		# to superimpose the A chains
		U = self.get_rotation_matrix_Kabsh(xyz_contact_decoy,xyz_contact_ref)

		# rotate the entire fragment
		xyz_contact_decoy = transform.rotation_matrix(xyz_contact_decoy,U,center=False)

		# compute the RMSD
		irmsd = error * self.get_rmsd(xyz_contact_decoy,xyz_contact_ref)

		# export the pdb for verifiactions
		if exportpath is not None:

			# update the sql database
			sql_decoy.update_xyz(xyz_contact_decoy,index=index_contact_decoy)
			sql_ref.update_xyz(xyz_contact_ref,index=index_contact_ref)

			sql_decoy.exportpdb(exportpath+'/irmsd_decoy.pdb',index=index_contact_decoy)
			sql_ref.exportpdb(exportpath+'/irmsd_ref.pdb',index=index_contact_ref)

		# close the db
		sql_decoy.close()
		sql_ref.close()

		return irmsd

	# compute only Fnat
	def compute_Fnat(self):

		# create the sql
		sql_decoy = pdb2sql(self.decoy,sqlfile='mol1.db')
		sql_ref = pdb2sql(self.ref,sqlfile='mol2.db')

		# get the contact atoms
		cutoff = 10.
		contact_pairs_decoy = sql_decoy.get_contact_atoms(cutoff=cutoff,
			                                              extend_to_residue=False,
			                                              only_backbone_atoms=True,
			                                              return_contact_pairs=True)


		contact_pairs_ref   = sql_ref.get_contact_atoms(cutoff=cutoff,
			                                            extend_to_residue=False,
			                                            only_backbone_atoms=True,
			                                            return_contact_pairs=True)


		# get the data 
		data_decoy = sql_decoy.get('chainID,resSeq,name')
		data_ref = sql_ref.get('chainID,resSeq,name')

		# form the pair data
		data_pair_decoy = []
		for indA,indexesB in contact_pairs_decoy.items():
			data_pair_decoy += [  [data_decoy[indA],data_decoy[indB]] for indB in indexesB   ]

		# form the pair data
		data_pair_ref = []
		for indA,indexesB in contact_pairs_ref.items():
			data_pair_ref += [  [data_ref[indA],data_ref[indB]] for indB in indexesB   ]

		# count the number of pairs 
		# of the ref present in the decoy
		count = 0
		for refpair in data_pair_ref:
			if refpair in data_pair_decoy:
				count += 1

		# normalize
		Fnat = count/len(data_pair_ref)

		return Fnat


	@staticmethod
	def compute_DockQScore(Fnat,lrmsd,irmsd,d1=8.5,d2=1.5):

		def scale_rms(rms,d):
			return(1./(1+(rms/d)**2))

		return 1./3 * (  Fnat + scale_rms(lrmsd,d1) + scale_rms(irmsd,d2) ) 


	@staticmethod
	def get_identical_atoms(db1,db2,chain):

		# get data
		data1 = db1.get('chainID,resSeq,name',chain=chain)
		data2 = db2.get('chainID,resSeq,name',chain=chain)

		# tuplify
		data1 = [tuple(d1) for d1 in data1]
		data2 = [tuple(d2) for d2 in data2]

		# get the intersection
		shared_data = list(set(data1).intersection(data2))

		# get the xyz
		xyz1,xyz2 = [],[]
		for data in shared_data:
			query = 'SELECT x,y,z from ATOM WHERE chainID=? AND resSeq=? and name=?'
			xyz1.append(list(list(db1.c.execute(query,data))[0]))
			xyz2.append(list(list(db2.c.execute(query,data))[0]))

		return xyz1,xyz2


	@staticmethod 
	def get_rmsd(P,Q):
		n = len(P)
		return np.sqrt(1./n*np.sum((P-Q)**2))


	@staticmethod 
	def align_Kabsch(P,Q):

		# translate the points
		P = transform.translation(P,get_trans_vect(P))
		Q = transform.translation(Q,get_trans_vect(Q))

		# get the matrix
		U = self.get_rotation_matrix_Kabsh(P,Q)

		# form the new ones
		P = np.dot(U,P.T).T

		return P,Q

	@staticmethod 
	def get_trans_vect(P):
		return  -np.mean(P,0)

	@staticmethod 		
	def get_rotation_matrix_Kabsh(P,Q):

		pshape = P.shape
		qshape = Q.shape

		if pshape[0] == qshape[0]:
			npts = pshape[0]
		else:
			print("Matrix don't have the same number of points")
			print(P.shape,Q.shape)
			sys.exit()


		p0,q0 = np.abs(np.mean(P,0)),np.abs(np.mean(Q,0))
		eps = 1E-6
		if any(p0 > eps) or any(q0 > eps):
			print('Center the fragments first')
			print(p0,q0)
			sys.exit()


		# form the covariance matrix
		A = np.dot(P.T,Q)/npts

		# SVD the matrix
		V,S,W = np.linalg.svd(A)

		# the W matrix returned here is
		# already its transpose
		# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.svd.html
		W = W.T

		# determinant 
		d = np.linalg.det(np.dot(W,V.T))

		# form the U matrix
		Id = np.eye(3)
		if d < 0:
			Id[2,2] = -1

		U = np.dot(W,np.dot(Id,V.T))

		return U



if __name__ == '__main__':

	#test_points()

	BM4 = '/home/nico/Documents/projects/deeprank/data/HADDOCK/BM4_dimers/'
	mol1 = BM4 + 'decoys_pdbFLs/1AK4/water/1AK4_100w.pdb'
	mol2 = BM4 + 'decoys_pdbFLs/1AK4/water/1AK4_1w.pdb'

	CAPRI = '/home/nico/Documents/projects/deeprank/data/CAPRI/'
	decoy = CAPRI + 'T29/complex_0024.pdb'
	#decoy = CAPRI + 'T29/2vdu_chainFD.pdb'
	ref = CAPRI + 'T29/2vdu_chainEB.pdb'

	sim = StructureSimilarity(decoy,ref)

	lrmsd = sim.compute_lrmsd(exportpath='./')
	irmsd = sim.compute_irmsd(exportpath='./')
	Fnat = sim.compute_Fnat()
	dockQ = sim.compute_DockQScore(Fnat,lrmsd,irmsd)
	print('L-RMSD = ', lrmsd )
	print('I-RMSD = ', irmsd )
	print('Fnat   = ', Fnat  )
	print('DockQ  = ', dockQ )