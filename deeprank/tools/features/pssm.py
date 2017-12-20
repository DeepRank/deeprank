import os
from deeprank.tools import NaivePSSM
#####################################################################################
#
#	THE MAIN FUNCTION CALLED IN THE INTERNAL FEATURE CALCULATOR
#
#####################################################################################

def __compute_feature__(pdb_data,featgrp):

	path = os.path.dirname(os.path.realpath(__file__))
	PSSM = path + '/PSSM/'

	mol_name = os.path.split(featgrp.name)[0]
	mol_name = mol_name.lstrip('/')

	pssm = NaivePSSM(mol_name,pdb_data,PSSM)

	# get the sasa info
	pssm.get_sasa()

	# read the raw data
	pssm.read_PSSM_data()

	# get the pssm smoothed sum score
	pssm.process_pssm_data()

	# get the feature vales
	pssm.get_feature_value()
	
	# export in the hdf5 file
	pssm.export_dataxyz_hdf5(featgrp)
