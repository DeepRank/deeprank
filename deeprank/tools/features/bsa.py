from deeprank.tools import BSA
import  os

#####################################################################################
#
#	THE MAIN FUNCTION CALLED IN THE INTERNAL FEATURE CALCULATOR
#
#####################################################################################

def __compute_feature__(pdb_data,featgrp,featgrp_raw):

	# create the BSA instance
	bsa = BSA(pdb_data)

	# get the structure/calc 
	bsa.get_structure()

	# get the feature info
	bsa.get_contact_residue_sasa()

	# export in the hdf5 file
	bsa.export_dataxyz_hdf5(featgrp)
	bsa.export_data_hdf5(featgrp_raw)

	# close the file
	bsa.sql.close()