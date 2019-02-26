from deeprank.features import AtomicFeature
import numpy as np
import pkg_resources
import unittest


# in case you change the ref don't forget to:
# - comment the first line (E0=1)
# - uncomment the last two lines (Total = ...)
# - use the corresponding PDB file to test
#REF = './1AK4/atomic_features/ref_1AK4_100w.dat'
pdb = './decoys/complex.1.pdb'
#pdb = './1AK4_10w.pdb'
test_name = './atomic_features/test_2OUL_1.dat'

# get the force field included in deeprank
# if another FF has been used to compute the ref
# change also this path to the correct one
FF = pkg_resources.resource_filename('deeprank.features','') + '/forcefield/'

# declare the feature calculator instance
atfeat = AtomicFeature(pdb,fix_chainID=True,
                       param_charge = FF + 'protein-allhdg5-4_new.top',
                       param_vdw    = FF + 'protein-allhdg5-4_new.param',
                       patch_file   = FF + 'patch.top')
# assign parameters
atfeat.assign_parameters()

# only compute the pair interactions here
atfeat.evaluate_pair_interaction(save_interactions=test_name,print_interactions=True)


# # make sure that the other properties are not crashing
# atfeat.compute_coulomb_interchain_only(contact_only=True)
# atfeat.compute_coulomb_interchain_only(contact_only=False)

# # make sure that the other properties are not crashing
# atfeat.compute_vdw_interchain_only(contact_only=True)
# atfeat.compute_vdw_interchain_only(contact_only=False)

# # close the db
# atfeat.sqldb.close()



