#!/usr/bin/env python

import os
from time import time

import numpy as np

from cleandata import *
from deeprank.generate import *

##########################################################################
#
#   GENERATE THE DATA BASE AT ONCE
#   --> assemble the pdbs
#   --> compute the features on the fly
#   --> compute the targets on the fly
#   --> map the features on the grid
#
##########################################################################

# adress of the BM4 folder
BM4 = ''


def generate(LIST_NAME, clean=False):

    for NAME in LIST_NAME:

        print(NAME)
        # sources to assemble the data base
        pdb_source = [BM4 + 'decoys_pdbFLs/' + NAME + '/water/']
        pdb_native = [BM4 + 'BM4_dimers_bound/pdbFLs_ori']

        # init the data assembler
        database = DataGenerator(
            pdb_source=pdb_source,
            pdb_native=pdb_native,
            data_augmentation=None,
            compute_targets=[
                'deeprank.targets.dockQ',
                'deeprank.targets.binary_class'],
            compute_features=[
                'deeprank.features.AtomicFeature',
                'deeprank.features.FullPSSM',
                'deeprank.features.PSSM_IC',
                'deeprank.features.BSA',
                'deeprank.features.ResidueDensity'],
            hdf5=NAME + '.hdf5',
        )

        if not os.path.isfile(database.hdf5):
            t0 = time()
            print('{:25s}'.format('Create new database') + database.hdf5)
            database.create_database()
            print(' ' * 25 + '--> Done in %f s.' % (time() - t0))
        else:
            print('{:25s}'.format('Use existing database') + database.hdf5)

        # map the features
        # grid_info = {
        #     'number_of_points' : [30,30,30],
        #     'resolution' : [1.,1.,1.],
        #     'atomic_densities' : {'CA':3.5,'CB':3.5,'N':3.5,'O':3.5,'C':3.5},
        # }

        # t0 =time()
        # print('{:25s}'.format('Map features in database') + database.hdf5)
        # database.map_features(grid_info,time=False,try_sparse=True,cuda=True,gpu_block=[8,8,8])
        # print(' '*25 + '--> Done in %f s.' %(time()-t0))

        # clean the data file
        if clean:
            t0 = time()
            print('{:25s}'.format('Clean datafile') + database.hdf5)
            clean_dataset(database.hdf5)
            print(' ' * 25 + '--> Done is %f s.' % (time() - t0))


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description='launch multiple HDF5 calculations')
    parser.add_argument(
        '-s',
        '--status',
        action='store_true',
        help='Only list the directory')
    parser.add_argument(
        '-d',
        '--device',
        help="GPU device to use",
        default='1',
        type=str)
    parser.add_argument(
        '-m',
        '--mol',
        nargs='+',
        help='name of the molecule to process',
        default=None,
        type=str)
    parser.add_argument(
        '-i',
        '--init',
        help="index of the first molecule to process",
        default=0,
        type=int)
    parser.add_argument(
        '-f',
        '--final',
        help="index of the last molecule to process",
        default=0,
        type=int)
    parser.add_argument(
        '--clean',
        help="Clean the datafiles",
        action='store_true')
    args = parser.parse_args()

    # get the names of the directories
    names = np.sort(os.listdir(BM4 + 'decoys_pdbFLs/')).tolist()

    # remove some files
    # as stated in the README some complex don't have a water stage
    remove_file = ['README', '2H7V', '1F6M', '1ZLI', '1IBR', '1R8S', '1Y64']
    for r in remove_file:
        names.remove(r)

    # get the names of thehdf5 already there
    hdf5 = list(filter(lambda x: '.hdf5' in x, os.listdir()))
    status = ['Done' if n + '.hdf5' in hdf5 else '' for n in names]
    size = [
        "{:5.2f}".format(
            os.path.getsize(
                n +
                '.hdf5') /
            1E9) if n +
        '.hdf5' in hdf5 else '' for n in names]

    # list the dir and their status
    if args.status:
        print(
            '\n' +
            '=' *
            50 +
            '\n= Current status of the datase \n' +
            '=' *
            50)
        for i, (n, s, w) in enumerate(zip(names, status, size)):
            if w == '':
                print('% 4d: %6s %5s %s' % (i, n, s, w))
            else:
                print('% 4d: %6s %5s %s GB' % (i, n, s, w))
        print('-' * 50)
        print(
            ': Status  --> %4.3f %%  done' % (status.count('Done') / len(status) * 100))
        print(': Mem Tot --> %4.3f GB\n' % sum(list(map(lambda x: float(x),
                                                        filter(lambda x: len(x) > 0, size)))))

    # compute the data
    else:

        if args.mol is not None:
            MOL = args.mol
        else:
            MOL = names[args.init:args.final + 1]

        # set the cuda device
        #os.environ['CUDA_DEVICE'] = args.device

        # generate the data
        generate(MOL, clean=args.clean)
