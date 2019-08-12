import os
import warnings

import numpy as np

from deeprank.features import FeatureClass
from deeprank.generate import settings
from deeprank.tools import pdb2sql


def printif(string, cond): return print(string) if cond else None


########################################################################
#
#   Definition of the class
#
########################################################################

class FullPSSM(FeatureClass):

    def __init__(self, mol_name=None, pdb_file=None, pssm_path=None,
                 pssm_format='new', out_type='pssmvalue'):
        """Compute all the PSSM data.

            Simply extracts all the PSSM information and
            store that into features

        Args:
            mol_name (str): name of the molecule. Defaults to None.
            pdb_file (str): name of the pdb_file. Defaults to None.
            pssm_path (str): path to the pssm data. Defaults to None.
            pssm_format (str): "old" or "new" pssm format.
                Defaults to 'new'.
            out_type (str): which feature to generate, 'pssmvalue' or 'pssmic'.
                 Defaults to 'pssmvalue'. 'pssm_format' must be 'new'
                 when set type is 'pssmic'.

        Examples:
            >>> path = '/home/test/PSSM_newformat/'
            >>> pssm = FullPSSM(mol_name='2ABZ',
            >>>                pdb_file='2ABZ_1w.pdb',
            >>>                pssm_path=path)
            >>> pssm.read_PSSM_data()
            >>> pssm.get_feature_value()
            >>> print(pssm.feature_data_xyz)
        """

        super().__init__("Residue")

        self.mol_name = mol_name
        self.pdb_file = pdb_file
        self.pssm_path = pssm_path
        self.ref_mol_name = self.get_ref_mol_name(mol_name)
        self.pssm_format = pssm_format
        self.out_type = out_type.lower()

        if isinstance(pdb_file, str) and mol_name is None:
            self.mol_name = os.path.splitext(pdb_file)[0]

        if self.out_type == 'pssmic' and not self.pssm_format == 'new':
            raise ValueError(f"You must provide 'new' format PSSM files"
                             f" to generate PSSM IC features.")

        if self.out_type == 'pssmvalue':
            # the residue order in res_names must be consistent with
            # that in PSSM file
            res_names = ('ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                         'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                         'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                         'SER', 'THR', 'TRP', 'TYR', 'VAL')
            self.feature_names = tuple(['PSSM_' + n for n in res_names])

            for name in self.feature_names:
                self.feature_data[name] = {}
                self.feature_data_xyz[name] = {}
        else:
            name = 'pssm_ic'
            self.feature_names = (name,)
            self.feature_data[name] = {}
            self.feature_data_xyz[name] = {}

    @staticmethod
    def get_ref_mol_name(mol_name):
        """Get the bared mol name."""
        return mol_name.split('_')[0]

    def read_PSSM_data(self):
        """Read the PSSM data into a dictionary."""

        names = os.listdir(self.pssm_path)
        fnames = list(filter(lambda x: self.ref_mol_name in x, names))
        num_pssm_files = len(fnames)

        if num_pssm_files == 0:
            raise FileNotFoundError(
                f'No PSSM file found for '
                f'{self.mol_name} in {self.pssm_path}')

        # old format with one file for all chains
        # and only pssm data
        if self.pssm_format == 'old':

            if num_pssm_files > 1:
                raise ValueError(
                    f'Multiple PSSM files found for '
                    f'{self.mol_name} in {self.pssm_path}')
            else:
                fname = fnames[0]

            with open(os.path.join(self.pssm_path, fname), 'rb') as f:
                data = f.readlines()
            raw_data = list(map(lambda x: x.decode('utf-8').split(), data))

            self.pssm_res_id = np.array(raw_data)[:, :3]
            self.pssm_res_id = [(r[0], int(r[1]), r[2])
                                for r in self.pssm_res_id]
            self.pssm_data = np.array(raw_data)[:, 3:].astype(np.float)
            """
            pssm_res_id: [('B', 573, 'HIS'), (...)]
            pssm_data: [[...], [...]]
            """

        # new format with 2 files (each chain has one file)
        # and aligned mapping and IC (i.e. the iScore format)
        elif self.pssm_format == 'new':

            if num_pssm_files < 2:
                raise FileNotFoundError(
                    f'Only one PSSM file found for '
                    f'{self.mol_name} in {self.pssm_path}')

            # get chain name
            fnames.sort()
            chain_names = [n.split('.')[1] for n in fnames]

            resmap = {
                'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP',
                'C': 'CYS', 'E': 'GLU', 'Q': 'GLN', 'G': 'GLY',
                'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
                'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER',
                'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
                'B': 'ASX', 'U': 'SEC', 'Z': 'GLX'
            }

            iiter = 0
            for chainID, fn in zip(chain_names, fnames):

                with open(os.path.join(self.pssm_path, fn), 'rb') as f:
                    data = f.readlines()
                raw_data = list(
                    map(lambda x: x.decode('utf-8').split(), data))

                rd = np.array(raw_data)[1:, :2]
                rd = [(chainID, int(r[0]), resmap[r[1]]) for r in rd]
                if self.out_type == 'pssmvalue':
                    pd = np.array(raw_data)[1:, 4:-1].astype(np.float)
                else:
                    pd = np.array(raw_data)[1:, -1].astype(np.float)
                    pd = pd.reshape(pd.shape[0], -1)

                if iiter == 0:
                    self.pssm_res_id = rd
                    self.pssm_data = pd
                    iiter = 1

                else:
                    self.pssm_res_id += rd
                    self.pssm_data = np.vstack((self.pssm_data, pd))

        self.pssm = dict(zip(self.pssm_res_id, self.pssm_data))

    def get_feature_value(self, cutoff=5.5):
        """get the feature value."""

        sql = pdb2sql(self.pdb_file)

        # set achors for all residues and get their xyz
        xyz_info = sql.get('chainID,resSeq,resName', name='CB')
        xyz_info += sql.get('chainID,resSeq,resName', name='CA',
                            resName='GLY')

        xyz = sql.get('x,y,z', name='CB')
        xyz += sql.get('x,y,z', name='CA', resName='GLY')

        xyz_dict = {}
        for pos, info in zip(xyz, xyz_info):
            xyz_dict[tuple(info)] = pos

        # get interface contact residues
        # ctc_res = ([chain 1 residues], [chain2 residues])
        ctc_res = sql.get_contact_residue(cutoff=cutoff)
        sql.close()
        ctc_res = ctc_res[0] + ctc_res[1]

        # handle with small interface or no interface
        total_res = len(ctc_res)
        if total_res == 0:
            raise ValueError(
                f"No interface residue found with the cutoff {cutoff}Å."
                f" Failed to calculate the features of FullPSSM/PSSM_IC")
        elif total_res < 5:  # this is an empirical value
            warnings.warn(
                f"Only {total_res} interface residues found with "
                f"cutoff {cutoff}Å. Be careful with using the features "
                f" FullPSSM/PSSM_IC")

        # check if interface residues have pssm values
        ctc_res_set = set(ctc_res)
        pssm_res_set = set(self.pssm.keys())
        if len(ctc_res_set.intersection(pssm_res_set)) == 0:
            raise ValueError(
                f"All interface residues have no pssm values."
                f"Check residue chainID/ID/name consistency "
                f"between PDB and PSSM files"
            )
        elif len(ctc_res_set.difference(pssm_res_set)) > 0:
            ctc_res_wo_pssm = ctc_res_set.difference(pssm_res_set)
            ctc_res_with_pssm = ctc_res_set - ctc_res_wo_pssm
            warnings.warn(
                f"The following interface residues have "
                f" no pssm value:\n {ctc_res_wo_pssm}"
            )
        else:
            ctc_res_with_pssm = ctc_res

        # get feature values
        for res in ctc_res_with_pssm:
            chain = {'A': 0, 'B': 1}[res[0]]
            key = tuple([chain] + xyz_dict[res])
            for name, value in zip(self.feature_names, self.pssm[res]):
                """
                Make sure the feature_names and pssm[res] have
                consistent order of the 20 residue types
                name: PSSM_ALA
                value: -3.0
                res: ('B', 573, 'HIS')
                key: (0, -19.346, 6.156, -3.44)
                """
                self.feature_data[name][res] = [value]
                self.feature_data_xyz[name][key] = [value]


#####################################################################################
#
#   THE MAIN FUNCTION CALLED IN THE INTERNAL FEATURE CALCULATOR
#
#####################################################################################


def __compute_feature__(pdb_data, featgrp, featgrp_raw, out_type='pssmvalue'):

    if settings.__PATH_PSSM_SOURCE__ is None:
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path,  'PSSM_NEW')
    else:
        path = settings.__PATH_PSSM_SOURCE__

    mol_name = os.path.split(featgrp.name)[0]
    mol_name = mol_name.lstrip('/')

    pssm = FullPSSM(mol_name, pdb_data, path, out_type=out_type)

    # read the raw data
    pssm.read_PSSM_data()

    # get the feature vales
    pssm.get_feature_value()

    # export in the hdf5 file
    pssm.export_dataxyz_hdf5(featgrp)
    pssm.export_data_hdf5(featgrp_raw)


########################################################################
#
#   IF WE JUST TEST THE CLASS
#
########################################################################
if __name__ == '__main__':

    from time import time
    t0 = time()
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pdb_file = os.path.join(base_path, "test/1AK4/native/1AK4.pdb")
    path = os.path.join(base_path, "test/1AK4/pssm_new")
    # pssm = FullPSSM(mol_name='1AK4', pdb_file=pdb_file, pssm_path=path,
    #                 pssm_format='new', out_type='pssmic')
    pssm = FullPSSM(mol_name='1AK4', pdb_file=pdb_file, pssm_path=path,
                    pssm_format='new', out_type='pssmvalue')

    # get the pssm smoothed sum score
    pssm.read_PSSM_data()
    pssm.get_feature_value()
    print(pssm.feature_data)
    print()
    print(pssm.feature_data_xyz)
    print(' Time %f ms' % ((time()-t0)*1000))
