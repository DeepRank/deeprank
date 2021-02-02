import itertools
import warnings
import pdb2sql

from deeprank.features import FeatureClass
from deeprank import config


class ResidueDensity(FeatureClass):

    def __init__(self, pdb_data, chain1='A', chain2='B'):
        """Compute the residue contacts between polar/apolar/charged residues.

        Args:
            pdb_data (list(byte) or str): pdb data or pdb filename
            chain1 (str): First chain ID. Defaults to 'A'
            chain2 (str): Second chain ID. Defaults to 'B'

        Example:
            >>> rcd = ResidueDensity('1EWY_100w.pdb')
            >>> rcd.get(cutoff=5.5)
            >>> rcd.extract_features()
        """

        self.pdb_data = pdb_data
        self.sql = pdb2sql.interface(pdb_data)
        self.chains_label = [chain1, chain2]
        self.chain1 = chain1
        self.chain2 = chain2

        self.feature_data = {}
        self.feature_data_xyz = {}

        self.residue_types = config.AA_properties

    def get(self, cutoff=5.5):
        """Get residue contacts.

        Raises:
            ValueError: No residue contact found.
        """
        # res = {('chain1,resSeq,resName'): set(
        #                               ('chain2,res1Seq,res1Name),
        #                               ('chain2,res2Seq,res2Name'))}
        res = self.sql.get_contact_residues(chain1=self.chains_label[0],
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
            total_ctc += self.residue_contacts[i].density['total']
        total_ctc = total_ctc / 2

        # handle with small interface or no interface
        if total_ctc == 0:
            # first close the sql
            self.sql._close()

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
        """Compute the feature of residue contacts between polar/apolar/charged
        residues.

        It generates following features:     RCD_apolar-apolar
        RCD_apolar-charged     RCD_charged-charged     RCD_polar-apolar
        RCD_polar-charged     RCD_polar-polar     RCD_total
        """

        self.feature_data['RCD_total'] = {}  # raw data for human read
        self.feature_data_xyz['RCD_total'] = {}  # for machine read

        restype = ['polar', 'apolar', 'charged']
        pairtype = ['-'.join(p) for p in
                    list(itertools.combinations_with_replacement(restype, 2))]
        for p in pairtype:
            self.feature_data['RCD_' + p] = {}
            self.feature_data_xyz['RCD_' + p] = {}

        for key, res in self.residue_contacts.items():

            # total density in raw format
            self.feature_data['RCD_total'][key] = [res.density['total']]

            # get the center
            _, xyz = self.get_residue_center(self.sql, res=key)
            xyz_key = tuple([{self.chain1: 0, self.chain2: 1}[key[0]]] + xyz[0])

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

def __compute_feature__(pdb_data, featgrp, featgrp_raw, chain1, chain2):
    """Main function called in deeprank for the feature calculations.

    Args:
        pdb_data (list(bytes)): pdb information
        featgrp (str): name of the group where to save xyz-val data
        featgrp_raw (str): name of the group where to save human readable data
        chain1 (str): First chain ID
        chain2 (str): Second chain ID
    """

    # create instance
    resdens = ResidueDensity(pdb_data, chain1=chain1, chain2=chain2)

    # get the residue conacts
    resdens.get()

    # extract the features
    resdens.extract_features()

    # export in the hdf5 file
    resdens.export_dataxyz_hdf5(featgrp)
    resdens.export_data_hdf5(featgrp_raw)

    # close sql
    resdens.sql._close()

########################################################################
#
#  TEST THE CLASS
#
########################################################################


if __name__ == '__main__':

    import os
    from pprint import pprint

    # get base path */deeprank, i.e. the path of deeprank package
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))))
    pdb_file = os.path.join(base_path, "test/1AK4/native/1AK4.pdb")

    # create instance
    resdens = ResidueDensity(pdb_file)

    resdens.get()
    resdens.extract_features()
    resdens.sql._close()

    pprint(resdens.feature_data)
    print()
    pprint(resdens.feature_data_xyz)
