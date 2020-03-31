import numpy as np


class FeatureClass(object):

    def __init__(self, feature_type):
        """Master class from which all the other feature classes should be
        derived.

            Each subclass must compute:

            - self.feature_data: dictionary of features in
                human readable format, e.g.

                for atomic features:
                {'coulomb': data_dict_clb, 'vdwaals': data_dict_vdw}
                    data_dict_clb = {atom_info: [values]}
                        atom_info = (chainID, resSeq, resName, name)

                for residue features:
                {'PSSM_ALA': data_dict_pssmALA, ...}
                    data_dict_pssmALA = {residue_info: [values]}
                        residue_info = (chainID, resSeq, resName, name)

            - self.feature_data_xyz: dictionary of features in
                xyz-val format, e.g.

                {'coulomb': data_dict_clb, 'vdwaals': data_dict_vdw}
                    data_dict_clb = {xyz_info: [values]}
                        xyz_info = (chainNum, x, y, z)

        Args:
            feature_type(str): 'Atomic' or 'Residue'
        """
        self.type = feature_type
        self.feature_data = {}
        self.feature_data_xyz = {}

    def export_data_hdf5(self, featgrp):
        """Export the data in human readable format to HDF5's group.

        - For atomic features, the format of the data must be:
            {(chainID, resSeq, resName, name): [values]}
        - For residue features, the format must be:
            {(chainID, resSeq, resName): [values]}
        """
        # loop through the datadict and name
        for name, data in self.feature_data.items():

            ds = []
            for key, value in data.items():

                # residue based feature
                if len(key) == 3:

                    # tags
                    feat = '{:>4}{:>10}{:>10}'.format(key[0], key[1], key[2])

                # atomic based features
                elif len(key) == 4:

                    # tags
                    feat = '{:>4}{:>10}{:>10}{:>10}'.format(
                        key[0], key[1], key[2], key[3])

                # values
                # note that feature_raw values have low precision
                for v in value:
                    feat += '    {: 1.6E}'.format(v)

                # append
                ds.append(feat)

            if ds:
                ds = np.array(ds).astype('|S' + str(len(ds[0])))
            else:
                ds = np.array(ds)


            # create the dataset
            if name + '_raw' in featgrp:
                old_data = featgrp[name + '_raw']
                old_data[...] = ds
            else:
                featgrp.create_dataset(name + '_raw', data=ds)

    ########################################
    #
    # export the data in an HDF5 file group
    # the format of the data is here
    # PRO : fast when mapping
    # CON : only usefull for deeprank
    #
    ########################################

    def export_dataxyz_hdf5(self, featgrp):
        """Export the data in xyz-val format in an HDF5 file group.

        For atomic and residue the format of the data must be:
        {(chainNum(0 or 1), x, y, z): [values]}
        """

        # loop through the datadict and name
        for name, data in self.feature_data_xyz.items():

            # create the data set
            ds = np.array([list(key) + value for key, value in data.items()])

            # create the dataset
            if name in featgrp:
                old = featgrp[name]
                old[...] = ds
            else:
                featgrp.create_dataset(name, data=ds)
