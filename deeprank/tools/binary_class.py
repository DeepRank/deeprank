import numpy as np
import os

def __compute_target__(decoy,outdir='./'):

	mol_name = decoy.split('/')[-1][:-4]
	export_file = outdir + '/' + mol_name

	# get the ref
	ref = os.path.dirname(os.path.realpath(decoy)) + '/ref.pdb'

	# if the two files are the same
	if os.path.getsize(decoy) == os.path.getsize(ref):
		if open(decoy,'r').read() == open(ref,'r').read():
			binclass = 1
	else:
		binclass = 0
	
	np.savetxt(export_file + '.BINCLASS',np.array([binclass]),fmt='%d')
