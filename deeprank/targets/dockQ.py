import warnings

import numpy as np

from deeprank.targets import capri
from deeprank.tools.StructureSimilarity import StructureSimilarity


def __compute_target__(decoy, targrp):
    """calculate DOCKQ.

    Args:
        decoy(str): pdb data of the decoy
        targrp(hdf5 group): HDF5 'targets' group
    """

    tarname = 'DOCKQ'
    if tarname in targrp.keys():
        del targrp[tarname]
        warnings.warn(f"Removed old {tarname} from {targrp.parent.name}")

    irmsd = capri.__compute_target__(decoy, targrp, "IRMSD")
    lrmsd = capri.__compute_target__(decoy, targrp, "LRMSD")
    fnat = capri.__compute_target__(decoy, targrp, "FNAT")

    dockQ = StructureSimilarity.compute_DockQScore(fnat, lrmsd, irmsd)
    targrp.create_dataset(tarname, data=np.array(dockQ))
