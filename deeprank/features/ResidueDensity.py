import itertools
import warnings

from deeprank.features import FeatureClass
from deeprank.tools import pdb2sql


class ResidueDensity(FeatureClass):

    def __init__(self, pdb_data, chainA='A', chainB='B'):
        """Compute the residue contacts between 
           polar/apolar/charged residues.

        Args :
            pdb_data (list(byte) or str): pdb data or pdb filename
            chainA (str, optional): name of the first chain
            chainB (str, optional): name of the second chain

        Example :
        >>> rcd = ResidueDensity('1EWY_100w.pdb')
        >>> rcd.get(cutoff=5.5)
        >>> rcd.extract_features()
        """

        self.pdb_data = pdb_data
        self.sql = pdb2sql(pdb_data)
        self.chains_label = [chainA, chainB]

        self.feature_data = {}
        self.feature_data_xyz = {}

        self.residue_types = {'CYS': 'polar', 'HIS': 'polar',
                              'ASN': 'polar', 'GLN': 'polar',
                              'SER': 'polar', 'THR': 'polar',
                              'TYR': 'polar', 'TRP': 'polar',
                              'ALA': 'apolar', 'PHE': 'apolar',
                              'GLY': 'apolar', 'ILE': 'apolar',
                              'VAL': 'apolar', 'MET': 'apolar',
                              'PRO': 'apolar', 'LEU': 'apolar',
                              'GLU': 'charged', 'ASP': 'charged',
                              'LYS': 'charged', 'ARG': 'charged'}

    def get(self, cutoff=5.5):
        """Get residue contacts

        Raises:
            ValueError: No residue contact found.
        """
        # res = {('chainA,resSeq,resName'): set(
        #                               ('chainB,res1Seq,res1Name),
        #                               ('chainB,res2Seq,res2Name'))}
        res = self.sql.get_contact_residue(chain1=self.chains_label[0],
                                           chain2=self.chains_label[1],
                                           cutoff=cutoff,
                                           return_contact_pairs=True)

        self.residue_contacts = {}
        for key, other_res in res.items():
            # some residues are not amino acids
            if key[2] not in self.residue_types:
                continue

            if key not in self.residue_contacts:
                self.residue_contacts[key] = residue_pair(
                    key, self.residue_types[key[2]])
            self.residue_contacts[key].density['total'] += len(other_res)

            for key2 in other_res:

                # some residues are not amino acids
                if key2[2] not in self.residue_types:
                    continue

                self.residue_contacts[key].density[
                    self.residue_types[key2[2]]] += 1
                self.residue_contacts[key].connections[
                    self.residue_types[key2[2]]].append(key2)

                if key2 not in self.residue_contacts:
                    self.residue_contacts[key2] = residue_pair(
                        key2, self.residue_types[key2[2]])

                self.residue_contacts[key2].density['total'] += 1
                self.residue_contacts[key2].density[
                    self.residue_types[key[2]]] += 1
                self.residue_contacts[key2].connections[
                    self.residue_types[key[2]]].append(key)

        # calculate the total number of contacts
        total_ctc = 0
        for i in self.residue_contacts:
            total_ctc += self.residue_contacts[i].density['total'][()]
        total_ctc = total_ctc/2

        # handle with small interface or no interface
        if total_ctc == 0:
            # first close the sql
            self.sql.close()

            raise ValueError(
                f"No residue contact found with the cutoff {cutoff}Å. "
                f"Failed to calculate the feature residue contact "
                f"density.")

        elif total_ctc < 5:  # this is an empirical value
            warnings.warn(
                f"Only {total_ctc} residue contacts found with "
                f" cutoff {cutoff}Å. Be careful with using the feature "
                f"residue contact density")

    def extract_features(self):
        """Compute the feature of residue contacts between 
            polar/apolar/charged residues.

            It generates following features:
                RCD_apolar-apolar
                RCD_apolar-charged
                RCD_charged-charged
                RCD_polar-apolar
                RCD_polar-charged
                RCD_polar-polar
                RCD_total
        """

        self.feature_data['RCD_total'] = {}  # raw data for human read
        self.feature_data_xyz['RCD_total'] = {}  # for machine read

        restype = ['polar', 'apolar', 'charged']
        pairtype = ['-'.join(p) for p in
                    list(itertools.combinations_with_replacement(restype, 2))]
        for p in pairtype:
            self.feature_data['RCD_'+p] = {}
            self.feature_data_xyz['RCD_'+p] = {}

        for key, res in self.residue_contacts.items():

            # total density in raw format
            self.feature_data['RCD_total'][key] = [res.density['total']]

            # get the type of the center
            atcenter = 'CB'
            if key[2] == 'GLY':
                atcenter = 'CA'

            # get the xyz of the center atom
            xyz = self.sql.get(
                'x,y,z', resSeq=key[1], chainID=key[0], name=atcenter)[0]
            #xyz = np.mean(self.sql.get('x,y,z',resSeq=key[1],chainID=key[0]),0).tolist()

            xyz_key = tuple([{'A': 0, 'B': 1}[key[0]]] + xyz)
            self.feature_data_xyz['RCD_total'][xyz_key] = [
                res.density['total']]

            # iterate through all the connection
            for r in restype:
                pairtype = 'RCD_' + res.type + '-' + r
                if pairtype not in self.feature_data:
                    pairtype = 'RCD_' + r + '-' + res.type
                self.feature_data[pairtype][key] = [res.density[r]]
                self.feature_data_xyz[pairtype][xyz_key] = [res.density[r]]


class residue_pair(object):

    def __init__(self, res, rtype):
        """Ancillary class that holds information for a given residue."""

        self.res = res
        self.type = rtype
        self.density = {'total': 0, 'polar': 0, 'apolar': 0, 'charged': 0}
        self.connections = {'polar': [], 'apolar': [], 'charged': []}


########################################################################
#
#   THE MAIN FUNCTION CALLED IN THE INTERNAL FEATURE CALCULATOR
#
########################################################################

def __compute_feature__(pdb_data, featgrp, featgrp_raw):

    # create instance
    resdens = ResidueDensity(pdb_data)

    # get the residue conacts
    resdens.get()

    # extract the features
    resdens.extract_features()

    # export in the hdf5 file
    resdens.export_dataxyz_hdf5(featgrp)
    resdens.export_data_hdf5(featgrp_raw)

    # close sql
    resdens.sql.close()
