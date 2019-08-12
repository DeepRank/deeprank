from deeprank.features.FullPSSM import FullPSSM
from deeprank.features.FullPSSM import __compute_feature__ as func

########################################################################
#
#   Definition of the class
#
########################################################################


class PSSM_IC(FullPSSM):
    pass

#####################################################################################
#
#   THE MAIN FUNCTION CALLED IN THE INTERNAL FEATURE CALCULATOR
#
#####################################################################################


def __compute_feature__(pdb_data, featgrp, featgrp_raw):

    func(pdb_data, featgrp, featgrp_raw, out_type='pssmic')


#####################################################################################
#
#   IF WE JUST TEST THE CLASS
#
#####################################################################################


if __name__ == '__main__':

    from time import time
    import os
    t0 = time()
    # get base path */deeprank, i.e. the path of deeprank package
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pdb_file = os.path.join(base_path, "test/1AK4/native/1AK4.pdb")
    path = os.path.join(base_path, "test/1AK4/pssm_new")
    pssmic = FullPSSM('1AK4', pdb_file, path, out_type='pssmic')

    # get the pssm smoothed sum score
    pssmic.read_PSSM_data()
    pssmic.get_feature_value()
    print(pssmic.feature_data)
    print()
    print(pssmic.feature_data_xyz)
    print()
    print(' Time %f ms' % ((time()-t0)*1000))
