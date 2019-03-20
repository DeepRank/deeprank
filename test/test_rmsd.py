import unittest
import numpy as np
from deeprank.tools import StructureSimilarity
import os

class TestStructureSimilarity(unittest.TestCase):
    """Test StructureSimialrity."""

    @staticmethod
    def test_rmsd():
        """Compute IRMSD/LRMSD and comapre with ProFIT generated values."""

        # specify wich data to us
        MOL = './1AK4/'
        decoys = MOL + '/decoys/'
        ref    = MOL + '/native/1AK4.pdb'
        data   = MOL + '/haddock_data/'

        # get the list of decoy names
        decoy_list = [decoys+'/'+n for n in list(filter(lambda x: '.pdb' in x, os.listdir(decoys)))]

        # reference data used to compare ours
        haddock_data = {}
        haddock_files =  [data+'1AK4.Fnat',data+'1AK4.lrmsd',data+'1AK4.irmsd']

        # extract the data from the haddock files
        for i,fname in enumerate(haddock_files):

            # read the file
            f = open(fname,'r')
            data = f.readlines()
            data = [d.split() for d in data if not d.startswith('#')]
            f.close()

            # extract/store the data
            for line in data:
                mol_name = line[0].split('.')[0]
                if i == 0:
                    haddock_data[mol_name] = np.zeros(3)
                haddock_data[mol_name][i] = float(line[1])


        # init all the data handlers
        nconf = len(haddock_data)
        deep = np.zeros((nconf,3))
        hdk = np.zeros((nconf,3))

        # compute the data with deeprank
        deep_data = {}
        for i,decoy in enumerate(decoy_list):

            sim = StructureSimilarity(decoy,ref)
            lrmsd = sim.compute_lrmsd_fast(method='svd',lzone='1AK4.lzone')
            irmsd = sim.compute_irmsd_fast(method='svd',izone='1AK4.izone')
            fnat = sim.compute_Fnat_fast(ref_pairs='1AK4.refpairs')

            mol_name = decoy.split('/')[-1].split('.')[0]
            deep_data[mol_name] =  [fnat,lrmsd,irmsd]
            deep[i,:] = deep_data[mol_name]
            hdk[i,:] = haddock_data[mol_name]

        # print the deltas
        delta = np.max(np.abs(deep-hdk),0)

        # assert the data
        if not np.all(delta<[1E-3,1,1E-3]):
            raise AssertionError()


    @staticmethod
    def test_slow():
        """Compute IRMSD/LRMSD from pdb2sql methd to make sure it doesn't crash."""

        # specify wich data to us
        MOL = './1AK4/'
        decoy = MOL + '/decoys/1AK4_cm-it0_745.pdb'
        ref    = MOL + '/native/1AK4.pdb'

        sim = StructureSimilarity(decoy,ref)
        trash = sim.compute_lrmsd_pdb2sql(method='svd')
        trash = sim.compute_irmsd_pdb2sql(method='svd')
        trash = sim.compute_Fnat_pdb2sql()
        print(trash)


    def setUp(self):
        """Setup the test by removing old files."""

        files = ['1AK4.lzone','1AK4.izone','1AK4.refpairs']
        for f in files:
            if os.path.isfile(f):
                os.remove(f)

if __name__ == '__main__':
    unittest.main()

