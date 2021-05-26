from pdb2sql import pdb2sql
import numpy


def __compute_feature__(pdb_data, featgrp, featgrp_raw, mutant):

    pdb = pdb2sql(mutant.pdb_path)

    try:
        chain_ids = sorted(set(pdb.get("chainID")))
        chain_numbers = {chain_id: index for index, chain_id in enumerate(chain_ids)}

        data = numpy.array([[chain_numbers[chain_id], x, y, z, x + y + z]
                            for chain_id, x, y, z in pdb.get("chainID,x,y,z")])
        featgrp.create_dataset("feature2", data=data)

    finally:
        pdb._close()
