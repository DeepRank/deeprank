from deeprank.tools import AtomicFeature
import os

#####################################################################################
#
#	THE MAIN FUNCTION CALLED IN THE INTERNAL FEATURE CALCULATOR
#
#####################################################################################

def __compute_feature__(pdb_data,featgrp,featgrp_raw):

	path = os.path.dirname(os.path.realpath(__file__))
	FF = path + '/forcefield/'

	atfeat = AtomicFeature(pdb_data,
		                   param_charge = FF + 'protein-allhdg5-4_new.top',
						   param_vdw    = FF + 'protein-allhdg5-4_new.param',
						   patch_file   = FF + 'patch.top')

	atfeat.assign_parameters()

	# only compute the pair interactions here
	atfeat.evaluate_pair_interaction(print_interactions=False)

	# compute the charges
	# here we extand the contact atoms to
	# entire residue containing at least 1 contact atom
	atfeat.evaluate_charges(extend_contact_to_residue=True)

	# export in the hdf5 file
	atfeat.export_dataxyz_hdf5(featgrp)
	atfeat.export_data_hdf5(featgrp_raw)

	# close
	atfeat.sqldb.close()


	