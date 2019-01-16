import os
import numpy as np

class FeatureClass(object):

    def __init__(self,feature_type):

        ''' Master class fron which all the other Feature classes should be derived."""

        Each subclass must compute :

        - self.feature_data : dictionary of features in human readable format
           e.g : {'coulomb':data_dict_clb[(atom info):value]
                  'vdwaals':data_dict_vdw[(atom info):value]  }

        - self.feature_data_xyz : dictionary of feature in xyz-val format
           e.g : {'coulomb':data_dict_clb[(chainID atom xyz):value]
                  'vdwaals':data_dict_vdw[(chainID atom xyz):value]  }

        Args:
            feature_type (str): 'Atomic' or 'Residue'

        '''

        self.type = feature_type
        self.feature_data = {}
        self.feature_data_xyz = {}
        self.export_directories = {}

    def export_data_hdf5(self,featgrp):
        """Export the data in human readable format in an HDF5 file group.

        - For **atomic features**, the format of the data must be : chainID  resSeq resNum name [values]
        - For **residue features**, the format must be : chainID  resSeq resNum [values]

        """

        # loop through the datadict and name
        for name,data in self.feature_data.items():

            ds = []
            for key,value in data.items():

                # residue based feature
                if len(key) == 3:

                    # tags
                    feat = '{:>4}{:>10}{:>10}'.format(key[0],key[1],key[2])

                # atomic based features
                elif len(key) == 4:

                    # tags
                    feat = '{:>4}{:>10}{:>10}{:>10}'.format(key[0],key[1],key[2],key[3])

                # values
                for v in value:
                    feat += '    {: 1.6E}'.format(v)

                # append
                ds.append(feat)

            # put in the hdf5 file
            ds = np.array(ds).astype('|S'+str(len(ds[0])))

            # create the dataset
            if name+'_raw' in featgrp:
                old_data = featgrp[name+'_raw']
                old_data[...] = ds
            else:
                featgrp.create_dataset(name+'_raw',data=ds)



    ########################################
    #
    # export the data in an HDF5 file group
    # the format of the data is here
    #
    # for atomic and residue features
    # x y z [values]
    #
    # PRO : fast when mapping
    # CON : only usefull for deeprank
    #
    ########################################
    def export_dataxyz_hdf5(self,featgrp):
        """Export the data in xyz-val format in an HDF5 file group.

        For **atomic** and **residue** the format of the data must be :  x y z [values]
        """


        # loop through the datadict and name
        for name,data in self.feature_data_xyz.items():

            # create the data set
            ds = np.array([list(key)+value for key,value in data.items()])

            # create the dataset
            if name in featgrp:
                old = featgrp[name]
                old[...] = ds
            else:
                featgrp.create_dataset(name,data=ds)