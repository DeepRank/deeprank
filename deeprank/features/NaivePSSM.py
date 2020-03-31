import os
from time import time

import numpy as np
import pdb2sql

from deeprank.features import FeatureClass
from deeprank.tools import SASA


def printif(string, cond): return print(string) if cond else None


##########################################################################
#
#   Definition of the class
#
##########################################################################

class NaivePSSM(FeatureClass):

    def __init__(
            self,
            mol_name=None,
            pdbfile=None,
            pssm_path=None,
            nmask=17,
            nsmooth=3,
            debug=False):
        """Compute compressed PSSM data.

        The method is adapted from:
        Simplified Sequence-based method for ATP-binding prediction using contextual local evolutionary conservation
        Algorithms for Molecular Biology 2014 9:7

        Args:
            mol_name (str): name of the molecule
            pdbfile (str): name of the dbfile
            pssm_path (str): path to the pssm data
            nmask (int, optional):
            nsmooth (int, optional):

        Example:

        >>> path = '/home/nico/Documents/projects/deeprank/data/HADDOCK/BM4_dimers/PSSM_newformat/'
        >>> pssm = NaivePSSM(mol_name = '2ABZ', pdbfile='2ABZ_1w.pdb',pssm_path=path)
        >>>
        >>> # get the surface accessible solvent area
        >>> pssm.get_sasa()
        >>>
        >>> # get the pssm smoothed sum score
        >>> pssm.read_PSSM_data()
        >>> pssm.process_pssm_data()
        >>> pssm.get_feature_value()
        >>> print(pssm.feature_data_xyz)
        """

        super().__init__("Residue")
        print("== Warning : Please don't use NaivePSSM as a feature it's very experimental")

        self.mol_name = mol_name
        self.pdbfile = pdbfile
        self.pssm_path = pssm_path
        self.molname = self.get_mol_name(mol_name)
        self.nmask = nmask
        self.nsmooth = nsmooth
        self.debug = debug

        if isinstance(pdbfile, str) and mol_name is None:
            self.mol_name = os.path.splitext(pdbfile)[0]

    def get_sasa(self):
        """Get the sasa of the residues."""
        sasa = SASA(self.pdbfile)
        self.sasa = sasa.neighbor_vector()

    @staticmethod
    def get_mol_name(mol_name):
        """Get the bared mol name."""
        return mol_name.split('_')[0]

    def read_PSSM_data(self):
        """Read the PSSM data."""

        names = os.listdir(self.pssm_path)
        fname = [n for n in names if n.find(self.molname) == 0]

        if len(fname) > 1:
            raise ValueError(
                'Multiple PSSM files found for %s in %s',
                self.mol_name,
                self.pssm_path)
        if len(fname) == 0:
            raise FileNotFoundError(
                'No PSSM file found for %s in %s',
                self.mol_name,
                self.pssm_path)
        else:
            fname = fname[0]

        f = open(self.pssm_path + '/' + fname, 'rb')
        data = f.readlines()
        f.close()
        raw_data = list(map(lambda x: x.decode('utf-8').split(), data))

        self.res_data = np.array(raw_data)[:, :3]
        self.res_data = [(r[0], int(r[1]), r[2]) for r in self.res_data]
        self.pssm_data = np.array(raw_data)[:, 3:].astype(np.float)

    def process_pssm_data(self):
        """Process the PSSM data."""

        self.pssm_data = self._mask_pssm(self.pssm_data, nmask=self.nmask)
        self.pssm_data = self._filter_pssm(self.pssm_data)
        self.pssm_data = self._smooth_pssm(
            self.pssm_data, msmooth=self.nsmooth)
        self.pssm_data = np.mean(self.pssm_data, 1)

    @staticmethod
    def _mask_pssm(pssm_data, nmask=17):

        nres = len(pssm_data)

        masked_pssm = np.copy(pssm_data)
        for idata in range(nres):
            istart = np.max([idata - nmask, 0])
            iend = np.min([idata + nmask + 1, nres])
            N = 1. / (2 * (iend - 1 - istart))
            masked_pssm[idata, :] -= N * np.sum(pssm_data[istart:iend, :], 0)
        return masked_pssm

    @staticmethod
    def _filter_pssm(pssm_data):
        pssm_data[pssm_data <= 0] = 0
        return pssm_data

    @staticmethod
    def _smooth_pssm(pssm_data, msmooth=3):

        nres = len(pssm_data)
        smoothed_pssm = np.copy(pssm_data)
        for idata in range(nres):
            istart = np.max([idata - msmooth, 0])
            iend = np.min([idata + msmooth + 1, nres])
            N = 1. / (2 * (iend - 1 - istart))
            smoothed_pssm[idata, :] = N * np.sum(pssm_data[istart:iend, :], 0)
        return smoothed_pssm

    def get_feature_value(self, contact_only=True):
        """get the feature value."""

        sql = pdb2sql.interface(self.pdbfile)
        xyz_info = sql.get('chainID,resSeq,resName', name='CB')
        xyz = sql.get('x,y,z', name='CB')

        xyz_dict = {}
        for pos, info in zip(xyz, xyz_info):
            xyz_dict[tuple(info)] = pos

        contact_residue = sql.get_contact_residue(cutoff=5.5)
        contact_residue = contact_residue["A"] + contact_residue["B"]
        sql._close()

        pssm_data_xyz = {}
        pssm_data = {}

        for res, data in zip(self.res_data, self.pssm_data):

            if contact_only and res not in contact_residue:
                continue

            if tuple(res) in xyz_dict:
                chain = {'A': 0, 'B': 1}[res[0]]
                key = tuple([chain] + xyz_dict[tuple(res)])
                sasa = self.sasa[tuple(res)]

                pssm_data[res] = [data * sasa]
                pssm_data_xyz[key] = [data * sasa]
            else:
                printif([tuple(res), ' not found in the pdbfile'], self.debug)

        # if we have no contact atoms
        if len(pssm_data_xyz) == 0:
            pssm_data_xyz[tuple([0, 0., 0., 0.])] = [0.0]
            pssm_data_xyz[tuple([1, 0., 0., 0.])] = [0.0]

        self.feature_data['pssm'] = pssm_data
        self.feature_data_xyz['pssm'] = pssm_data_xyz


##########################################################################
#
#   THE MAIN FUNCTION CALLED IN THE INTERNAL FEATURE CALCULATOR
#
##########################################################################

def __compute_feature__(pdb_data, featgrp, featgrp_raw):

    if '__PATH_PSSM_SOURCE__' not in globals():
        path = os.path.dirname(os.path.realpath(__file__))
        PSSM = path + '/PSSM/'
    else:
        PSSM = __PATH_PSSM_SOURCE__

    mol_name = os.path.split(featgrp.name)[0]
    mol_name = mol_name.lstrip('/')

    pssm = NaivePSSM(mol_name, pdb_data, PSSM)

    # get the sasa info
    pssm.get_sasa()

    # read the raw data
    pssm.read_PSSM_data()

    # get the pssm smoothed sum score
    pssm.process_pssm_data()

    # get the feature vales
    pssm.get_feature_value()

    # export in the hdf5 file
    pssm.export_dataxyz_hdf5(featgrp)
    pssm.export_data_hdf5(featgrp_raw)


##########################################################################
#
#   IF WE JUST TEST THE CLASS
#
##########################################################################
if __name__ == '__main__':

    t0 = time()
    path = '/home/nico/Documents/projects/deeprank/data/HADDOCK/BM4_dimers/PSSM_newformat/'
    pssm = NaivePSSM(mol_name='2ABZ', pdbfile='2ABZ_1w.pdb', pssm_path=path)

    # get the surface accessible solvent area
    pssm.get_sasa()

    # get the pssm smoothed sum score
    pssm.read_PSSM_data()
    pssm.process_pssm_data()
    pssm.get_feature_value()
    print(pssm.feature_data_xyz)
    print(' Time %f ms' % ((time() - t0) * 1000))
