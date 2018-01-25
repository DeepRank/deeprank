from deeprank.tools import ResidueDensity

#####################################################################################
#
#	THE MAIN FUNCTION CALLED IN THE INTERNAL FEATURE CALCULATOR
#
#####################################################################################

def __compute_feature__(pdb_data,featgrp,featgrp_raw):

	# create the BSA instance
	resdens = ResidueDensity(pdb_data)

	# get the densities
	resdens.get()

	# extract the features
	resdens.extract_features()

	# export in the hdf5 file
	resdens.export_dataxyz_hdf5(featgrp)
	resdens.export_data_hdf5(featgrp_raw)
