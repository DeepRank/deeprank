import os
import numpy as np

from time import time

from deeprank.tools import pdb2sql
from deeprank.tools import SASA
from deeprank.features import FeatureClass

printif = lambda string,cond: print(string) if cond else None


#####################################################################################
#
#   Definition of the class
#
#####################################################################################

class FullPSSM(FeatureClass):

    def __init__(self,mol_name=None,pdbfile=None,pssm_path=None,debug=False):

        '''Compute all the PSSM data.

       Simply extracts all the PSSM information and store that into features

        Args:
            mol_name (str): name of the molecule
            pdbfile (str): name of the dbfile
            pssm_path (str): path to the pssm data

        Example:

        >>> path = '/home/nico/Documents/projects/deeprank/data/HADDOCK/BM4_dimers/PSSM_newformat/'
        >>> pssm = FullPSSM(mol_name = '2ABZ', pdbfile='2ABZ_1w.pdb',pssm_path=path)
        >>>
        >>> # get the pssm smoothed sum score
        >>> pssm.read_PSSM_data()
        >>> pssm.get_feature_value()
        >>> print(pssm.feature_data_xyz)
        '''

        super().__init__("Residue")

        self.mol_name = mol_name
        self.pdbfile = pdbfile
        self.pssm_path = pssm_path
        self.molname = self.get_mol_name(mol_name)
        self.debug = debug

        if isinstance(pdbfile,str) and mol_name is None:
            self.mol_name = os.path.splitext(pdbfile)[0]

        res_names = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','LLE',
                     'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
        self.pssm_val_name = ['PSSM_' + n for n in res_names]

        for name in self.pssm_val_name:
            self.feature_data[name] = {}
            self.feature_data_xyz[name] = {}

    @staticmethod
    def get_mol_name(mol_name):
        """Get the bared mol name."""
        return mol_name.split('_')[0]

    def read_PSSM_data(self):
        """Read the PSSM data."""

        names = os.listdir(self.pssm_path)
        fname = [n for n in names if n.find(self.molname)==0]

        if len(fname)>1:
            raise ValueError('Multiple PSSM files found for %s in %s',self.pdbname,self.pssm_path)
        if len(fname)==0:
            raise FileNotFoundError('No PSSM file found for %s in %s',self.pdbname,self.pssm_path)
        else:
            fname = fname[0]

        f = open(self.pssm_path + '/' + fname,'rb')
        data = f.readlines()
        f.close()
        raw_data = list( map(lambda x: x.decode('utf-8').split(),data))

        self.res_data  = np.array(raw_data)[:,:3]
        self.res_data = [  (r[0],int(r[1]),r[2]) for r in self.res_data ]
        self.pssm_data = np.array(raw_data)[:,3:].astype(np.float)

    def get_feature_value(self,contact_only=True):
        """get the feature value."""

        sql = pdb2sql(self.pdbfile)
        xyz_info = sql.get('chainID,resSeq,resName',name='CB')
        xyz = sql.get('x,y,z',name='CB')

        xyz_dict = {}
        for pos,info in zip(xyz,xyz_info):
            xyz_dict[tuple(info)] = pos

        contact_residue = sql.get_contact_residue()
        contact_residue = contact_residue[0] + contact_residue[1]
        sql.close()

        pssm_data_xyz = {}
        pssm_data = {}

        for res,data in zip(self.res_data,self.pssm_data):

            if contact_only and res not in contact_residue:
                continue

            if tuple(res) in xyz_dict:
                chain = {'A':0,'B':1}[res[0]]
                key = tuple([chain] + xyz_dict[tuple(res)])

                for name,value in zip(self.pssm_val_name,data):
                    self.feature_data[name][res] = [value]
                    self.feature_data_xyz[name][key] = [value]

            else:
                printif([tuple(res), ' not found in the pdbfile'],self.debug)

        # if we have no contact atoms
        if len(pssm_data_xyz) == 0:
            pssm_data_xyz[tuple([0,0.,0.,0.])] = [0.0]
            pssm_data_xyz[tuple([1,0.,0.,0.])] = [0.0]

#####################################################################################
#
#   THE MAIN FUNCTION CALLED IN THE INTERNAL FEATURE CALCULATOR
#
#####################################################################################

def __compute_feature__(pdb_data,featgrp,featgrp_raw):

    if '__PATH_PSSM_SOURCE__' not in globals():
        path = os.path.dirname(os.path.realpath(__file__))
        PSSM = path + '/PSSM/'
    else:
        PSSM = __PATH_PSSM_SOURCE__

    mol_name = os.path.split(featgrp.name)[0]
    mol_name = mol_name.lstrip('/')

    pssm = FullPSSM(mol_name,pdb_data,PSSM)

    # read the raw data
    pssm.read_PSSM_data()

    # get the feature vales
    pssm.get_feature_value()

    # export in the hdf5 file
    pssm.export_dataxyz_hdf5(featgrp)
    pssm.export_data_hdf5(featgrp_raw)




#####################################################################################
#
#   IF WE JUST TEST THE CLASS
#
#####################################################################################

if __name__ == '__main__':

    t0 = time()
    path = '/home/nico/Documents/projects/deeprank/data/HADDOCK/BM4_dimers/PSSM_newformat/'
    pssm = FullPSSM(mol_name = '1AK4', pdbfile='1AK4_100w.pdb',pssm_path=path)


    # get the pssm smoothed sum score
    pssm.read_PSSM_data()
    pssm.get_feature_value()
    print(' Time %f ms' %((time()-t0)*1000))
