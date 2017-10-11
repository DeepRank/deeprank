import numpy as np
from deeprank.tools import transform, pdb2sql 
import sys

# compute the rmsd between two molecules
# based on https://en.wikipedia.org/wiki/Kabsch_algorithm
# P and Q are Nx3 matrices

def get_rmsd(P,Q):
	n = len(P)
	return np.sqrt(1./n*np.sum((P-Q)**2))

def align_Kabsch(P,Q):

	# translate the points
	P = transform.translation(P,get_trans_vect(P))
	Q = transform.translation(Q,get_trans_vect(Q))

	# get the matrix
	U = get_rotation_matrix_Kabsh(P,Q)

	# form the new ones
	P = np.dot(U,P.T).T

	return P,Q


def get_trans_vect(P):
	return  -np.mean(P,0)

	
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


def compute_lrmsd(decoy,ref,exportpath=None):

	'''
	Ref : DockQ: A Quality Measure for Protein-Protein Docking Models
	      http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0161879
	L-RMSD is computed by aligning the longest chain of the decoy to the one of the  reference
	and computing the RMSD of the shortest chain between decoy and ref
	'''

	# create teh sql
	sql_decoy = pdb2sql(decoy,sqlfile='mol1.db')
	sql_ref = pdb2sql(ref,sqlfile='mol2.db')

	# extract the pos of the dimer
	xyz_decoy = np.array(sql_decoy.get('x,y,z'))
	xyz_ref = np.array(sql_ref.get('x,y,z'))

	# extract the pos of chains A
	xyz_decoy_A = np.array(sql_decoy.get('x,y,z',chain='A'))
	xyz_ref_A = np.array(sql_ref.get('x,y,z',chain='A'))

	# extract the pos of chains B
	xyz_decoy_B = np.array(sql_decoy.get('x,y,z',chain='B'))
	xyz_ref_B = np.array(sql_ref.get('x,y,z',chain='B'))

	# detect which chain is the longest
	nA,nB = len(xyz_decoy_A),len(xyz_decoy_B)
	if nA>nB:
		xyz_decoy_long = xyz_decoy_A
		xyz_ref_long = xyz_ref_A
		chain_short = 'B'
	else:
		xyz_decoy_long = xyz_decoy_B
		xyz_ref_long = xyz_ref_B
		chain_short = 'A'

	# get the translation so that both A chains are centered
	tr_decoy = get_trans_vect(xyz_decoy_long)
	tr_ref = get_trans_vect(xyz_ref_long)

	# translate everything for 1
	xyz_decoy = transform.translation(xyz_decoy,tr_decoy)
	xyz_decoy_long = transform.translation(xyz_decoy_long,tr_decoy)

	# translate everuthing for 2
	xyz_ref = transform.translation(xyz_ref,tr_ref)
	xyz_ref_long = transform.translation(xyz_ref_long,tr_ref)

	# get the ideql rotation matrix
	# to superimpose the A chains
	U = get_rotation_matrix_Kabsh(xyz_decoy_long,xyz_ref_long)

	# rotate the entire fragment
	xyz_decoy = transform.rotation_matrix(xyz_decoy,U,center=False)

	# update the sql database
	sql_decoy.update_xyz(xyz_decoy)
	sql_ref.update_xyz(xyz_ref)

	# get the short chain pos 
	xyz_short_decoy = np.array(sql_decoy.get('x,y,z',chain=chain_short))
	xyz_short_ref = np.array(sql_ref.get('x,y,z',chain=chain_short))

	# compute the RMSD
	LRMSD =  get_rmsd(xyz_short_decoy,xyz_short_ref)

	# export the pdb for verifiactions
	if exportpath is not None:
		sql_decoy.exportpdb(exportpath+'/decoy_aligned.pdb')
		sql_ref.exportpdb(exportpath+'/ref_aligned.pdb')

	# close the db
	sql_decoy.close()
	sql_ref.close()

	return LRMSD


def compute_irmsd(decoy,ref,exportpath=None):

	'''
	Ref : DockQ: A Quality Measure for Protein-Protein Docking Models
	      http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0161879
	i-RMSD is computed selecting the back bone of the contact residue with a cutoff of 10A
	in the decoy. Align these as best as possible with their coutner part in the ref
	and and compute the RMSD
	'''

	# create thes sql
	sql_decoy = pdb2sql(decoy,sqlfile='mol1.db')
	sql_ref = pdb2sql(ref,sqlfile='mol2.db')

	# get the contact atoms
	cutoff = 10.
	contact_decoy = sql_decoy.get_contact_atoms(cutoff=cutoff,extend_to_residue=True,only_backbone_atoms=True)

	# make a single list
	index_contact_decoy = contact_decoy[0]+contact_decoy[1]


	# get the xyz and atom identifier of the decoy contact atoms
	xyz_contact_decoy = sql_decoy.get('x,y,z',index=index_contact_decoy)
	data_contact_decoy = sql_decoy.get('chainID,resSeq,name',index=index_contact_decoy)

	# get the xyz and atom indeitifier of the reference
	xyz_ref = sql_ref.get('x,y,z')
	data_ref = sql_ref.get('chainID,resSeq,name')

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
			print('Warning atom %s of res %d not found in reference' %(atom[1],atom[0]))
			xyz_contact_decoy[iat] = None
			index_contact_decoy[iat] = None
			clean_decoy = True

	# clean the xyz
	if clean_decoy:
		xyz_contact_decoy = [xyz for xyz in xyz_contact_decoy if xyz is not None]
		index_contact_decoy = [ind for ind in index_contact_decoy if ind is not None]

	# get the translation so that both A chains are centered
	tr_decoy = get_trans_vect(xyz_contact_decoy)
	tr_ref = get_trans_vect(xyz_contact_ref)

	# translate everything 
	xyz_contact_decoy = transform.translation(xyz_contact_decoy,tr_decoy)
	xyz_contact_ref   = transform.translation(xyz_contact_ref,tr_ref)

	# get the ideql rotation matrix
	# to superimpose the A chains
	U = get_rotation_matrix_Kabsh(xyz_contact_decoy,xyz_contact_ref)

	# rotate the entire fragment
	xyz_contact_decoy = transform.rotation_matrix(xyz_contact_decoy,U,center=False)

	# compute the RMSD
	IRMSD =  get_rmsd(xyz_contact_decoy,xyz_contact_ref)

	# export the pdb for verifiactions
	if exportpath is not None:

		# update the sql database
		sql_decoy.update_xyz(xyz_contact_decoy,index=index_contact_decoy)
		sql_ref.update_xyz(xyz_contact_ref,index=index_contact_ref)

		sql_decoy.exportpdb(exportpath+'/decoy_contact_residue_aligned.pdb',index=index_contact_decoy)
		sql_ref.exportpdb(exportpath+'/ref_contact_residue_aligned.pdb',index=index_contact_ref)

	# close the db
	sql_decoy.close()
	sql_ref.close()

	return IRMSD


def compute_Fnat(decoy,ref):

	# create the sql
	sql_decoy = pdb2sql(decoy,sqlfile='mol1.db')
	sql_ref = pdb2sql(ref,sqlfile='mol2.db')

	# get the contact atoms
	cutoff = 5.
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


def test_molecule(mol1,mol2):

	# create teh sql
	sql1 = pdb2sql(mol1,sqlfile='mol1.db')
	sql2 = pdb2sql(mol2,sqlfile='mol2.db')

	# extract the pos
	xyz1 = np.array(sql1.get('x,y,z'))
	xyz2 = np.array(sql2.get('x,y,z'))

	# extract the pos of chains A
	xyz1_A = np.array(sql1.get('x,y,z',chain='A'))
	xyz2_A = np.array(sql2.get('x,y,z',chain='A'))

	# get the translation so that both A chains are centered
	tr1 = get_trans_vect(xyz1_A)
	tr2 = get_trans_vect(xyz2_A)

	# translate everything for 1
	xyz1 = transform.translation(xyz1,tr1)
	xyz1_A = transform.translation(xyz1_A,tr1)

	# translate everuthing for 2
	xyz2 = transform.translation(xyz2,tr2)
	xyz2_A = transform.translation(xyz2_A,tr2)

	# get the ideql rotation matrix
	# to superimpose the A chains
	U = get_rotation_matrix_Kabsh(xyz1_A,xyz2_A)

	# rotate the entire fragment
	xyz1 = transform.rotation_matrix(xyz1,U,center=False)

	# print the rmsd
	print('RMSD = ', get_rmsd(xyz1,xyz2))

	sql1.update_xyz(xyz1)
	sql2.update_xyz(xyz2)

	sql1.exportpdb('mol1.pdb')
	sql2.exportpdb('mol2.pdb')

	sql1.close()
	sql2.close()

def test_points(npts=10,add_noise=False):

	# create two series of points
	P = np.random.rand(npts,3)
	Q = np.copy(P)

	if add_noise :
		eps = 1E-3
		Q += -eps+2*eps*np.random.rand(npts,3)

	# translate P
	trans = np.random.rand(3)
	P += trans

	# rotate P
	teta = 2*np.pi*np.random.rand()
	axis = -1 + 2*np.random.rand(3)
	axis /= np.linalg.norm(axis)
	P = transform.rotation_around_axis(P,axis,teta)

	# align P and Q
	P,Q = align_Kabsch(P,Q)

	# compute the rmsd
	rmsd = get_rmsd(P,Q)

	# output the result
	print('RMSD = ', rmsd)


if __name__ == '__main__':

	test_points()

	BM4 = '/home/nico/Documents/projects/deeprank/data/HADDOCK/BM4_dimers/'
	mol1 = BM4 + 'decoys_pdbFLs/1AK4/water/1AK4_100w.pdb'
	mol2 = BM4 + 'decoys_pdbFLs/1AK4/water/1AK4_1w.pdb'

	lrmsd = compute_lrmsd(mol1,mol2,exportpath='./')
	print('L-RMSD = ', lrmsd)

	irmsd = compute_irmsd(mol1,mol2,exportpath='./')
	print('I-RMSD = ', irmsd)

	Fnat = compute_Fnat(mol1,mol1)
	print('Fnat =',Fnat)