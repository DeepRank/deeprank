#!/usr/bin/env python
# Li Xue
# 20-Feb-2019 10:50
#
# This script takes input PDB files and PSSM files and outputs hdf5 files.
# Features (atomic and residue features) are calculated and mapped to 3D grids.
# Proteins are aligned. 


from deeprank.generate import *
from mpi4py import MPI
import sys
import os
import re
import glob
from time import time

comm = MPI.COMM_WORLD



"""
Some requirement of the naming of the files:
    1. case ID canNOT have underscore '_', e.g., '1ACB_CD'
    2. decoy file name should have this format: 2w83-AB_20.pdb (caseID_xxx.pdb)
    3. pssm file name should have this format: 2w83-AB.A.pssm (caseID.chainID.pssm or caseID.chainID.pdb.pssm)
"""
def fix_dir(ori):
    # '/home/lixue' => '/home/lixue/'
    # '/home//lixue' => '/home/lixue/'
    # '/home/lixue////' => '/home/lixue/'


    for idx,val in enumerate(ori):
        val = val.strip()

        if val =='':
            ori[idx] = val
            continue

        val = re.sub('/{2,}','/',val)
        val = re.sub('/$','', val)

        if val[-1] != '/':
            val = val + '/'
        ori[idx]= val

    return ori

def check_emptyPDBdir(ori_dirs):
    for val in ori_dirs:
        if not os.path.isdir(val):
            sys.exit(f"PDB Dir {val} does not exist.")
        if not glob.glob(f'{val}*.pdb'):
            sys.exit(f"{val} does not have pdb files.")

def check_emptyPSSMdir(ori_dirs):
    for val in ori_dirs:
        if not os.path.isdir(val):
            sys.exit(f"{val} does not exist.")
        if not glob.glob(f'{val}*.pssm'):
            sys.exit(f"{val} does not have pdb files.")

def main():
    if len(sys.argv) != 6:
        print(f"Usage: python {sys.argv[0]} caseID decoyDIR nativeDIR pssmDIR outDIR\n")
        sys.exit()

    caseID = sys.argv[1] # 1ACB, 1bj1-CF
    decoyDIR = sys.argv[2]
    nativeDIR= sys.argv[3]
    pssmDIR = sys.argv[4]
    outDIR = sys.argv[5]

    decoyDIR, nativeDIR, pssmDIR, outDIR = fix_dir([decoyDIR,nativeDIR,pssmDIR, outDIR])

    print(f"caseID: {caseID}")
    print(f"decoyDIR: {decoyDIR}")
    print(f"nativeDIR: {nativeDIR}")
    print(f"pssmDIR: {pssmDIR}")
    print (f"outDIR: {outDIR}")

    check_emptyPDBdir([decoyDIR, nativeDIR])
    check_emptyPSSMdir([pssmDIR])


    h5file = outDIR + caseID + '.hdf5'
    print(f"h5file: {h5file}")
    pdb_source     = [decoyDIR]
    pdb_native     = [nativeDIR] # pdb_native is only used to calculate i-RMSD, dockQ and so on. The native pdb files will not be saved in the hdf5 file
    pssm_source = pssmDIR

    sys.stdout.flush()

    if not os.path.isdir(outDIR):
        os.mkdir(outDIR)

    """Generate the database."""

    # clean old files
    #files = [h5file, h5file + '_norm.pckl']
    files1 = glob.glob(f"{outDIR}/*{h5file}")
    files2 = glob.glob(f"{outDIR}/*{h5file}_norm.pckl")
    files = files1 + files2
    for f in files:
        print (f"remove {f} if it exists")
        if os.path.isfile(f):
            os.remove(f)

    sys.stdout.flush()

    #init the data assembler

    database = DataGenerator(pdb_source= pdb_source,
                                pdb_native= pdb_native,
                                pssm_source= pssm_source,
                                align={"axis":'x','export':False},
                                #align={"selection":"interface","plane":"xy", 'export':False},
                                data_augmentation = None,
                                compute_targets  = ['deeprank.targets.dockQ','deeprank.targets.binary_class'],
                                compute_features = ['deeprank.features.AtomicFeature',
                                                    'deeprank.features.FullPSSM',
                                                    'deeprank.features.PSSM_IC',
                                                    'deeprank.features.BSA',
                                                    'deeprank.features.ResidueDensity'],
                                # check *.py in "features" folder for the features
                                hdf5=h5file,mpi_comm=comm)


    # compute features/targets, and write to hdf5 file
    print('{:25s}'.format('Create new database') + database.hdf5)
    database.create_database(prog_bar=True)


    # define the 3D grid
    grid_info = {
    'number_of_points' : [30,30,30],
    'resolution' : [1.,1.,1.],
    'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8},
    }

    # map the features to the 3D grid
    print('{:25s}'.format('Map features in database') + database.hdf5)
    database.map_features(grid_info,try_sparse=True, time=False, prog_bar=True)


if __name__ == '__main__':
    main()
