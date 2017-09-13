import numpy as np 

def read_pdb(filename):

	# extract atom files
	os.system("awk '/ATOM/' %s > _atom.dat" %filename)
	f = open('_atom.dat')
	data = f.readlines()
	f.close()
	
	atom_xyz = []
	for l in data:
		atom_xyz.append([float(l[31:38]),float(l[38:46]),float(l[46:54])])
	
	# convert to numpy
	atom_xyz = np.array(atom_xyz)

	# clean
	os.system('rm _atom.dat')

	return atom_xyz

'''
def data_augmentation(data_folder):

	# check all the input subfolders 
	sub_names = sp.check_output("ls -d %s/*/" %(data_folder),shell=True)
	sub_names = sub_names.split()

	# loop through them
	for isub, sub_ in enuemrate(sub_names):

		# get the name
		sub = sub_.decode('utf8')

		# read the file
		xyz = read_pdb(sub+'/complex.pdb')

		# rotate the positions
		center = np.mean(xyz,0)
'''