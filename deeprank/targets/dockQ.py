from deeprank.tools.StructureSimilarity import StructureSimilarity
import os
import numpy as np

def __compute_target__(decoy,targrp):

    # fet the mol group
    molgrp = targrp.parent
    molname = molgrp.name

    path = os.path.dirname(os.path.realpath(__file__))
    ZONE = path + '/zones/'

    if not os.path.isdir(ZONE):
        os.mkdir(ZONE)

    for target_name in ['LRMSD','IRMSD','FNAT','DOCKQ']:
        if target_name in targrp.keys():
            del targrp[target_name]

    # if we have a ref
    if '_' not in molname:

        # lrmsd = irmsd = 0 | fnat = dockq = 1
        print(f"WARNING: {molname} has no '_' indicating it is a bound complex. Assign 0, 0, 1 and 1 for lrmsd, irmsd, fnat, DockQ, respectively")
        targrp.create_dataset('LRMSD',data=np.array(0.0))
        targrp.create_dataset('IRMSD',data=np.array(0.0))
        targrp.create_dataset('FNAT', data=np.array(1.0))
        targrp.create_dataset('DOCKQ',data=np.array(1.0))

    # or it's a decoy
    else:

        # compute the izone/lzone/ref_pairs
        molname = molname.split('_')[0]
        lzone = ZONE + molname+'.lzone'
        izone = ZONE + molname+'.izone'
        ref_pairs = ZONE + molname + '.ref_pairs'

        # init the class
        decoy = molgrp['complex'][:]
        ref = molgrp['native'][:]
        sim = StructureSimilarity(decoy,ref)

        lrmsd = sim.compute_lrmsd_fast(method='svd',lzone=lzone)
        targrp.create_dataset('LRMSD',data=np.array(lrmsd))

        irmsd = sim.compute_irmsd_fast(method='svd',izone=izone)
        targrp.create_dataset('IRMSD',data=np.array(irmsd))

        Fnat = sim.compute_Fnat_fast(ref_pairs=ref_pairs)
        targrp.create_dataset('FNAT',data=np.array(Fnat))

        dockQ = sim.compute_DockQScore(Fnat,lrmsd,irmsd)
        targrp.create_dataset('DOCKQ',data=np.array(dockQ))

