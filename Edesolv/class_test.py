from pdb2sql import pdb2sql, interface
# a new class based on the FeatureClass
class CarbonAlphaFeature(FeatureClass):

    # init the class
    def __init__(self,pdbfile):
        super().__init__('Atomic')
            self.pdb = pdb

        # the feature extractor
        def get_feature(self):

            # create a sql database
            pdb_db = pdb2sql(self.pdb)
            db = interface(pdb_db)

            # get the contact atoms
            indA,indB = list(db.get_contact_atoms().values())
            contact = indA + indB

            # extract the atom keys and xyz of the contact CA atoms
            ca_keys = db.get('chainID,resName,resSeq,name',name='CA',rowID=contact)
            ca_xyz = db.get('x,y,z',name='CA',rowID=contact)

            # create the dictionary of human readable and xyz-val data
            hread, xyzval = {},{}
            for key,xyz in zip(ca_keys,ca_xyz):

                    # human readable
                    # { (chainID,resName,resSeq,name) : [val] }
                    hread[tuple(key)] = [1.0]

                    # xyz-val
                    # { (0|1,x,y,z) : [val] }
                    chain = [{'A':0,'B':1}[key[0]]]
                    k = tuple( chain + xyz)
                    xyzval[k] = [1.0]

            self.feature_data['CA'] = hread
            self.feature_data_xyz['CA'] = xyzval
