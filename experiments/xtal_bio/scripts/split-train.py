#!/usr/bin/env python

import math
import h5py
import re
import random
import os

__author__ = "Francesco Ambrosetti"
__email__ = "ambrosetti.francesco@gmail.com"


def read_caseIDS(f5_file):
    """Read a hdf5 file and return the list of
    keys with and without the rotation number (if present)
    f5_file = path to the hdf5 file"""

    with h5py.File(f5_file, 'r') as f5:
        caseIDs = list(f5.keys())
        codes = []
        for pdb in caseIDs:
            codes.append(re.match("\S+_\d+_\w", pdb).group())
        code_clean = list(set(codes))

    return caseIDs, code_clean


def get_proportions(bio_codes, xtal_codes, train_part=0.80):
    """Get the training and validation keys (no rotation) by keeping
    the two sets balanced (same/similar bio and xtal interfaces)
    bio_codes = list of codes of the bio interfaces
    xtal_codes = list of codes of the xtal interfaces
    train_part = percentage of the dataset (bio_codes + xtal_codes
                 to be used for training (default is 0.80)"""

    # Get train and validation proportions
    total = len(bio_codes) + len(xtal_codes)

    # Training set
    train_size = int(math.ceil(total * train_part))
    bio_train_size = int(train_size / 2)
    xtal_train_size = train_size - bio_train_size

    # Extract cases keeping valid and train balanced
    bio_train_cases = random.sample(bio_codes, bio_train_size)
    xtal_train_cases = random.sample(xtal_codes, xtal_train_size)

    train_cases = bio_train_cases + xtal_train_cases

    # Get validation cases
    all_codes = bio_codes + xtal_codes
    valid_cases = [x for x in all_codes if x not in train_cases]

    # Print everything
    print(f'Training set cases: {len(train_cases)}')
    print(f'Validation set cases: {len(valid_cases)}')

    return train_cases, valid_cases


def split_train(bio, xtal, train_part=0.80):
    """Get the training and validation keys (with rotations) by keeping
    the two sets balanced (same/similar bio and xtal interfaces)
    bio_codes = list of codes of the bio interfaces
    xtal_codes = list of codes of the xtal interfaces
    train_part = percentage of the dataset (bio_codes + xtal_codes
                 to be used for training (default is 0.80)"""

    # Read files
    all_cases_bio, bio_codes = read_caseIDS(bio)
    all_cases_xtal, xtal_codes = read_caseIDS(xtal)

    print(f'Bio interfaces: {len(bio_codes)}')
    print(f'Xtal interfaces: {len(xtal_codes)}')

    # Get train and validation IDs (without rotation)
    train_ids, valid_ids = get_proportions(bio_codes, xtal_codes, train_part)

    case_list = all_cases_bio + all_cases_xtal

    # Get the train IDs taking also the rotations
    train_cases = []
    for t in train_ids:
        train_cases = train_cases + list(filter(lambda x: t in x, case_list))

    # Sort IDs
    train_cases.sort()
    valid_ids.sort()

    print(f'Training set with rotation: {len(train_cases)}')
    print(f'Validation set without rotation: {len(valid_ids)}')

    return train_cases, valid_ids


def save_train_valid(bioh5, xtalh5, train_idx, val_idx, out_folder):
    """Save two subsets of the original hdf5 files for containing
    training and validation sets for both bio and xtal interfaces
    the output file names are: train_bio.hdf5, valid_bio.hdf5,
    train_xtal.hdf5 and valid_xtal.hdf5.

    bio_h5 = original hdf5 file for the bio interfaces
    xtal_h5 = original hdf5 file for the xtal interfaces
    train_idx = caseIDs for the training set (from split_train())
    val_idx = casedIDs for the validation set (from split_train())
    out_folder = path to the output folder"""

    # Create new hd5f files for the bio interfaces
    th5_bio = h5py.File(os.path.join(out_folder, 'train_bio.hdf5'), 'w')
    val5_bio = h5py.File(os.path.join(out_folder, 'valid_bio.hdf5'), 'w')
    bio_train = [x for x in train_idx if '_p' in x]
    bio_valid = [x for x in val_idx if '_p' in x]
    with h5py.File(bioh5, 'r') as f1:
        print('#### Creating Training  set file for bio ####')
        subset_h5(bio_train, f1, th5_bio)

        print('#### Creating Validation  set file for bio ####')
        subset_h5(bio_valid, f1, val5_bio)
    th5_bio.close()
    val5_bio.close()

    # Create new hd5f files for the xtal interfaces
    th5_xtal = h5py.File(os.path.join(out_folder, 'train_xtal.hdf5'), 'w')
    val5_xtal = h5py.File(os.path.join(out_folder, 'valid_xtal.hdf5'), 'w')
    xtal_train = [x for x in train_idx if '_n' in x]
    xtal_valid = [x for x in val_idx if '_n' in x]
    with h5py.File(xtalh5, 'r') as f1:
        print('#### Creating Training  set file for xtal ####')
        subset_h5(xtal_train, f1, th5_xtal)

        print('#### Creating Validation  set file for xtal ####')
        subset_h5(xtal_valid, f1, val5_xtal)
    th5_xtal.close()
    val5_xtal.close()


def subset_h5(idx, f1, f2):
    """Copy the selected keys (idx) of one hdf5 (f1) file into
     another hdf5 file (f2)
     idx = list of keys to copy
     f1 = handle of the first hdf5 file
     f2 = handle of the the second hdf5 file"""

    for b in idx:
        print(f'copying: {b}')

        # Get the name of the parent for the group we want to copy
        group_path = f1[b].parent.name

        # Check that this group exists in the destination file; if it doesn't, create it
        # This will create the parents too, if they don't exist
        group_id = f2.require_group(group_path)

        # Copy
        f1.copy(b, group_id)


if __name__ == '__main__':

    # Define input hdf5 files
    train_bio = '/projects/0/deepface/MANY/hdf5/manybio_au30.hdf5'
    train_xtal = '/projects/0/deepface/MANY/hdf5/manyxtal_au30.hdf5'

    # Get training and validation keys
    tr, va = split_train(train_bio, train_xtal, 0.80)

    # Save keys into files
    with open('train_caseIDS.txt', 'w') as x:
        for element in tr:
            x.write(element)
            x.write('\n')
    with open('valid_caseIDS.txt', 'w') as y:
        for ele in va:
            y.write(ele)
            y.write('\n')

    # Create training and validation hdf5 files
    save_train_valid(train_bio, train_xtal, tr, va, out_folder='/projects/0/deepface/MANY/hdf5/')
