import warnings

import numpy as np
from pdb2sql import StructureSimilarity

from deeprank.targets import rmsd_fnat


def __compute_target__(decoy, targrp):
    """calculate DOCKQ.

    Args:
        decoy(bytes): pdb data of the decoy
        targrp(hdf5 file handle): HDF5 'targets' group

    Examples:
        >>> f = h5py.File('1LFD.hdf5')
        >>> decoy = f['1LFD_9w/complex'][()]
        >>> targrp = f['1LFD_9w/targets']
    """

    tarname = 'DOCKQ'
    if tarname in targrp.keys():
        del targrp[tarname]
        warnings.warn(f"Removed old {tarname} from {targrp.parent.name}")

    irmsd = rmsd_fnat.__compute_target__(decoy, targrp, "IRMSD")
    lrmsd = rmsd_fnat.__compute_target__(decoy, targrp, "LRMSD")
    fnat = rmsd_fnat.__compute_target__(decoy, targrp, "FNAT")

    dockQ = StructureSimilarity.compute_DockQScore(fnat, lrmsd, irmsd)
    targrp.create_dataset(tarname, data=np.array(dockQ))
