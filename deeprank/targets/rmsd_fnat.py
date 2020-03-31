import os
import warnings

import numpy as np
from pdb2sql import StructureSimilarity


def __compute_target__(decoy, targrp, tarname, save_file=False):
    """Calcuate CAPRI metric IRMSD, LRMSD or FNAT.

    Args:
        decoy(bytes): pdb data of the decoy
        targrp(hdf5 file handle): HDF5 'targets' group
        tarnames(str): it must be IRMSD, LRMSD or FNAT.
        save_file(bool): save .izone, .lzone or .ref_pairs file or not,
            defaults to False.

    Returns:
        float: value of IRMSD, LRMSD or FNAT.

    Raises:
        ValueError: Wrong target name
        ValueError: native complex not exist
        ValueError: native complex has empty dataset

    Examples:
        >>> f = h5py.File('1LFD.hdf5')
        >>> decoy = f['1LFD_9w/complex'][()]
        >>> targrp = f['1LFD_9w/targets']
    """
    # check tarname
    tarname = tarname.upper()
    if tarname not in ("IRMSD", "LRMSD", "FNAT"):
        raise ValueError(f'Target name is wrong: {tarname}. '
                         f'It must be "IRMSD", "LRMSD", "FNAT"')

    # fet the mol group
    molgrp = targrp.parent
    molname = molgrp.name

    if save_file:
        path = os.path.dirname(os.path.realpath(__file__))
        ZONE = path + '/zones/'

        if not os.path.isdir(ZONE):
            os.mkdir(ZONE)

    if tarname in targrp.keys():
        del targrp[tarname]
        warnings.warn(f"Removed old {tarname} from {molname}")

    # if we have a ref
    if '_' not in molname:
        if tarname in ("IRMSD", "LRMSD"):
            target = 0.0
        elif tarname == "FNAT":
            target = 1.0

        targrp.create_dataset(tarname, data=np.array(target))
        warnings.warn(
            f"{molname} is a native/reference complex "
            f"without '_' in filename. Assign {target} for {tarname}")

    # or it's a decoy
    else:
        if 'native' not in molgrp:
            raise ValueError(
                f"'native' not exist for {molname}. "
                f"You must provide reference pdb for computing targets")
        elif molgrp['native'][()].shape is None:
            raise ValueError(
                f"'native' dataset is empty for {molname}. "
                f"You must provide reference pdb for computing targets")

        molname = molname.split('_')[0]

        # init the class
        decoy = molgrp['complex'][()]
        ref = molgrp['native'][()]
        sim = StructureSimilarity(decoy, ref)

        # comppute the izone/lzone/ref_pairs
        if tarname == "IRMSD":
            if save_file:
                izone = ZONE + molname + '.izone'
            else:
                izone = None
            target = sim.compute_irmsd_fast(method='svd', izone=izone)

        elif tarname == "LRMSD":
            if save_file:
                lzone = ZONE + molname + '.lzone'
            else:
                lzone = None
            target = sim.compute_lrmsd_fast(method='svd', lzone=lzone)

        elif tarname == "FNAT":
            target = sim.compute_fnat_fast()

        targrp.create_dataset(tarname, data=np.array(target))

    return target
