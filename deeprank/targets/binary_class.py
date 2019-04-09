import numpy as np

def __compute_target__(decoy, targrp):
    """
        pdb_data (bytes): PDB translated in bytes
        targrp (h5 file handle): name of the group where to store the targets

        e.g.,
        f = h5py.File('1LFD.hdf5')
        targrp = f['1LFD_9w/targets']

        list(targrp)
        ['DOCKQ', 'FNAT', 'IRMSD', 'LRMSD']
    """

    # fet the mol group
    molgrp = targrp.parent
    molname = molgrp.name

    for target_name in ['BIN_CLASS']:
        if target_name in targrp.keys():
            del targrp[target_name]

    if targrp['IRMSD'].value <=4:
        print (f"This is a hit (irmsd <=4A). {molname} -> irmsd: {targrp['IRMSD'].value}")
        classID = 1
    else:
        classID = 0

    targrp.create_dataset('BIN_CLASS',data=np.array(classID))

