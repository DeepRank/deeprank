"""
Merge content of multiple hdf5 files.

Usage: {0} t01.hdf5 t02.hdf5 out.hdf5
Author: {1}
"""
import sys
import h5py
import numpy as np

__author__ = 'CunliangGeng'
USAGE = __doc__.format(sys.argv[0], __author__)

def check_input(args):
    if len(args) <= 3:
        sys.exit(USAGE)

def merge_hdf5(h5ins, h5out):
    # /test/hit
    # /test/mol
    # /test/outputs
    # /test/targets

    if not isinstance(h5ins, list):
        raise ValueError('The parameter h5ins must be a list')

    hstack_groups = [
        '/train/hit',
        '/train/targets',
        '/valid/hit',
        '/valid/targets',
        '/test/hit',
        '/test/targets',]
    vstack_groups = [
        '/train/mol',
        '/train/outputs',
        '/valid/mol',
        '/valid/outputs',
        '/test/mol',
        '/test/outputs']
    test_groups = hstack_groups + vstack_groups

    # initialise output h5
    f5out = h5py.File(h5out, 'w')

    f5 = h5py.File(h5ins[0], 'r')
    epoch_groups = [i for i in f5.keys() if i.startswith('epoch')]

    for grp in epoch_groups:
        for subgrp in test_groups:
            f5.copy(grp+subgrp, f5out, grp+subgrp)
    f5.close()

    # merge h5 files
    for h5 in h5ins[1:]:
        f5 = h5py.File(h5, 'r')
        for grp in epoch_groups:
            for subgrp in hstack_groups:
                data = np.hstack((f5out[grp+subgrp][()], f5[grp+subgrp][()]))
                del f5out[grp+subgrp]
                f5out.create_dataset(grp+subgrp, data = data)
            for subgrp in vstack_groups:
                data = np.vstack((f5out[grp+subgrp][()], f5[grp+subgrp][()]))
                del f5out[grp+subgrp]
                if 'mol' in subgrp:
                    f5out.create_dataset(grp+subgrp, data = data.astype('S'))
                else:
                    f5out.create_dataset(grp+subgrp, data = data)

        f5.close()

    f5out.close()


if __name__ == "__main__":
    check_input((sys.argv[1:]))
    h5ins = sys.argv[1:-1]
    h5out = sys.argv[-1]
    # h5ins = [
    #     '/Users/clgeng/test/epoch_data_01.hdf5',
    #     '/Users/clgeng/test/epoch_data_03.hdf5',
    #     ]
    # h5out = '/Users/clgeng/test/tt.hdf5'
    merge_hdf5(h5ins, h5out)
