import numpy as np
from deeprank.tools import transform

#compute the rmsd between two molecules
# based on https://en.wikipedia.org/wiki/Kabsch_algorithm
# P and Q are Nx3 matrices

def get_rmsd(P,Q):
	n = len(P)
	return np.sqrt(1./n*np.sum((P-Q)**2))

def align_Kabsch(P,Q):

	pshape = P.shape
	qshape = Q.shape

	if pshape[0] == qshape[0]:
		npts = pshape[0]
	else:
		print("Matrix don't have the same number of points")
		sys.exit()

	# translate the coordinate to 
	P -= np.mean(P,0)
	Q -= np.mean(Q,0)

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

	# form the new ones
	P = np.dot(U,P.T).T

	return P,Q




if __name__ == '__main__':

	# create two series of points
	npts = 10
	eps = 1E-3
	P = np.random.rand(npts,3)
	Q = np.copy(P)
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

