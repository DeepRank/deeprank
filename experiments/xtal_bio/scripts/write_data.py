#!/usr/bin/env python
"""
Write out accuracy and loss values for each epoch

Usage: python {0} <deeprank training results in hdf5 format>  <output file name>
Example: python {0} epoch_data.hdf5 trail_01

Author: {1} ({2})
"""
import sys
import h5py
import numpy as np
import pandas as pd

__author__ = "Cunliang Geng"
__email__ = "gengcunliang AT gmail.com"
USAGE = __doc__.format(__file__, __author__, __email__)

def check_input(args):
    if len(args) != 2:
        sys.stderr.write(USAGE)
        sys.exit(1)

def out_acc(h5, fout):
    acc = np.vstack((h5['acc/train'][:],  h5['acc/valid'][:],  h5['acc/test'][:]))
    acc_df = pd.DataFrame(acc.transpose(), columns=['train', 'valid', 'test'])
    acc_df.to_csv('acc_'+fout+'.csv', index=False)

def out_loss(h5, fout):
    losses = np.vstack((h5['losses/train'][:],  h5['losses/valid'][:],  h5['losses/test'][:]))
    losses_df = pd.DataFrame(losses.transpose(), columns=['train', 'valid', 'test'])
    losses_df.to_csv('loss_'+fout+'.csv', index=False)

if __name__ == "__main__":
    check_input(sys.argv[1:])
    h5 = h5py.File(sys.argv[1], 'r')
    out_acc(h5, sys.argv[2])
    out_loss(h5, sys.argv[2])
