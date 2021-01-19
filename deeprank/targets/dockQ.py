import warnings

import numpy as np
from pdb2sql import StructureSimilarity

from deeprank.targets import rmsd_fnat


def __compute_target__(decoy, targrp):
    """Calculate DOCKQ.

    Args:
        decoy(bytes): pdb data of the decoy
        targrp(hdf5 file handle): HDF5 'targets' group
    """
    tarname = 'DOCKQ'
    if tarname in targrp.keys():
        del targrp[tarname]
        warnings.warn(f"Removed old {tarname} from {targrp.parent.name}")

    tarelems = ['FNAT', 'LRMSD', 'IRMSD']
    vals = {}
    for tarelem in tarelems:
        # if target element exist, then use its value; otherwise calculate it
        if tarelem not in targrp:
            _ = rmsd_fnat.__compute_target__(decoy, targrp, tarelem)
        # empty dataset
        elif targrp[tarelem][()].shape is None:
            _ = rmsd_fnat.__compute_target__(decoy, targrp, tarelem)

        # get the value
        vals[tarelem] = targrp[tarelem][()]

    dockQ = StructureSimilarity.compute_DockQScore(
            vals['FNAT'], vals['LRMSD'], vals['IRMSD'])
    targrp.create_dataset(tarname, data=np.array(dockQ))
