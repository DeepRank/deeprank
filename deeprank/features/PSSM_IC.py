from deeprank.features.FullPSSM import FullPSSM
from deeprank.features.FullPSSM import __compute_feature__ as func

########################################################################
#
#   Definition of the class
#
########################################################################


class PSSM_IC(FullPSSM):
    pass

##########################################################################
#
#   THE MAIN FUNCTION CALLED IN THE INTERNAL FEATURE CALCULATOR
#
##########################################################################

def __compute_feature__(pdb_data, featgrp, featgrp_raw, chain1, chain2):

    func(pdb_data, featgrp, featgrp_raw, chain1=chain1, chain2=chain2,
        out_type='pssmic')

##########################################################################
#
#   IF WE JUST TEST THE CLASS
#
##########################################################################


if __name__ == '__main__':

    from time import time
    from pprint import pprint
    import os

    t0 = time()
    # get base path */deeprank, i.e. the path of deeprank package
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))))
    pdb_file = os.path.join(base_path, "test/1AK4/native/1AK4.pdb")
    path = os.path.join(base_path, "test/1AK4/pssm_new")

    pssmic = FullPSSM('1AK4', pdb_file, path, out_type='pssmic')
    pssmic.read_PSSM_data()
    pssmic.get_feature_value()

    pprint(pssmic.feature_data)
    print()
    pprint(pssmic.feature_data_xyz)
    print()
    print(' Time %f ms' % ((time() - t0) * 1000))
