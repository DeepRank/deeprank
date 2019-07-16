#!/usr/bin/env python
# Li Xue
# 12-Jul-2019 21:13

# Count the hits.

import h5py

# def count_hits_from_input(hdf5_DIR = None):
#
#
# def main():
#
#     hdf5_DIR = '/home/lixue/DBs/BM5-haddock24/hdf5_withGridFeature'
#     count_hits_from_input(hdf5_DIR)

#h5FL = '/projects/0/deeprank/BM5/hdf5/hdf5_woGridFeature/000_1ACB.hdf5'
h5FL = '/projects/0/deeprank/BM5/hdf5/hdf5_withGridFeature/013_2ABZ.hdf5'

f = h5py.File(h5FL, 'r')

modelIDs = list(f)

for modelID in modelIDs:
    BIN_CLASS = f[modelID + '/targets/BIN_CLASS'][()]
    DOCKQ = f[modelID + '/targets/DOCKQ'][()]
    FNAT =  f[modelID + '/targets/FNAT'][()]
    IRMSD =  f[modelID + '/targets/IRMSD'][()]

    if IRMSD <=4:
        print(f"modelID: {modelID}, BIN: {BIN_CLASS}, irmsd: {IRMSD}")

