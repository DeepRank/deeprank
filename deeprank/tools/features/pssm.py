import os
from deeprank.tools import NaivePSSM
#####################################################################################
#
#	THE MAIN FUNCTION CALLED IN THE INTERNAL FEATURE CALCULATOR
#
#####################################################################################

def __compute_feature__(pdb_data,featgrp):

	path = os.path.dirname(os.path.realpath(__file__))
	PSSM = path + '/PSSM_newformat/'

	pssm = NaivePSSM(pdb_data,PSSM)

	# get the sasa info
	pssm.get_sasa()

	# get the pssm smoothed sum score
	pssm.read_PSSM_data()
	pssm.process_pssm_data()