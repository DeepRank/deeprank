import time
from deeprank.tools import atomicFeature
	
t0 = time.time()
PDB = 'complex.pdb'
FF = './forcefield/'
atfeat = atomicFeature(PDB,
	                   param_charge = FF + 'protein-allhdg5-4_new.top',
					   param_vdw    = FF + 'protein-allhdg5-4_new.param',
					   patch_file   = FF + 'patch.top')

atfeat.assign_parameters()

# only compute the pair interactions here
atfeat.evaluate_pair_interaction(print_interactions=True)

# compute the charges
# here we extand the contact atoms to
# entire residue containing at least 1 contact atom
atfeat.evaluate_charges(extend_contact_to_residue=True)

atfeat.export_data()
atfeat.sqldb.close()
print('Done in %f s' %(time.time()-t0))
