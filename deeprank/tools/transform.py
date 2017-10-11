import numpy as np

'''
This file contains several transformations of the
molecular coordinate that might be usefull during the 
definition of the data set.
'''

def translation(xyz,vect):
	return xyz + vect

def rotation_around_axis(xyz,axis,angle):

	# get the data
	ct,st = np.cos(angle),np.sin(angle)
	ux,uy,uz = axis

	# get the center of the molecule
	xyz0 = np.mean(xyz,0)

	# definition of the rotation matrix
	# see https://en.wikipedia.org/wiki/Rotation_matrix
	rot_mat = np.array([
	[ct + ux**2*(1-ct),			ux*uy*(1-ct) - uz*st,		ux*uz*(1-ct) + uy*st],
	[uy*ux*(1-ct) + uz*st,    	ct + uy**2*(1-ct),			uy*uz*(1-ct) - ux*st],
	[uz*ux*(1-ct) - uy*st,		uz*uy*(1-ct) + ux*st,   	ct + uz**2*(1-ct)   ]])

	# apply the rotation
	return np.dot(rot_mat,(xyz-xyz0).T).T + xyz0

		
def rotation_euler(xyz,alpha,beta,gamma):

	# precomte the trig
	ca,sa = np.cos(alpha),np.sin(alpha)
	cb,sb = np.cos(beta),np.sin(beta)
	cg,sg = np.cos(gamma),np.sin(gamma)


	# get the center of the molecule
	xyz0 = np.mean(xyz,0)

	# rotation matrices
	rx = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
	ry = np.array([[cb,0,sb],[0,1,0],[-sb,0,cb]])
	rz = np.array([[cg,-sg,0],[sg,cs,0],[0,0,1]])

	rot_mat = np.dot(rz,np.dot(ry,rz))

	# apply the rotation
	return np.dot(rot_mat,(xyz-xyz0).T).T + xyz0


def rotation_matrix(xyz,rot_mat,center=True):
	if center:
		xyz0 = np.mean(xyz)
		return np.dot(rot_mat,(xyz-xyz0).T).T + xyz0
	else:
		return np.dot(rot_mat,(xyz).T).T