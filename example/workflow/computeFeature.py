import os, sys
import numpy as np 
import subprocess as sp
from deeprank.tools import atomicFeature

# the root of the benchmark
BM4        = '/home/nico/Documents/projects/deeprank/data/HADDOCK/BM4_dimers/'
BM4        = sys.argv[1]

# dir for writing the data
dir_elec   = BM4 + 'ELEC/'
dir_vdw    = BM4 + 'VDW/'

# forcefield 
FF         = BM4 +'./forcefield/'

# conformation
decoys     = BM4 + '/decoys_pdbFLs/'
native     = BM4 + '/BM4_dimers_bound/pdbFLs_refined'

# filter the decoys
decoyID    = './decoyID.dat'

# get the names of all the decoy pdb files in the benchmark 
decoyName = sp.check_output('find %s -name  "*.pdb"' %decoys,shell=True).decode('utf-8').split()

# get the decoy ID we want to keep
decoyID = list(np.loadtxt(decoyID,str))

# filter the decy name
decoyName = [name for name in decoyName if name.split('/')[-1][:-4] in decoyID]


#get the natives names
nativeName = sp.check_output('find %s -name "*.pdb"' %native,shell=True).decode('utf-8').split()

# all the pdb we want
PDB_NAMES = nativeName + decoyName


# loop over the files
for PDB in PDB_NAMES:

	print('\nCompute Atomic Feature for %s' %(PDB.split('/')[-1][:-4]))
	atfeat = atomicFeature(
			 PDB,
             param_charge = FF + 'protein-allhdg5-4_new.top',
             param_vdw    = FF + 'protein-allhdg5-4_new.param',
             patch_file   = FF + 'patch.top',
             root_export  = BM4 )

	atfeat.assign_parameters()
	atfeat.evaluate_charges()
	atfeat.evaluate_pair_interaction(print_interactions=False)
	atfeat.export_data()
	atfeat.sqldb.close()



