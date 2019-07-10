import numpy as np
from deeprank.tools import pdb2sql
import os,pickle
_printif = lambda string,cond: print(string) if cond else None

class StructureSimilarity(object):

    def __init__(self,decoy,ref,verbose=False):
        """Compute the structure similarity between different molecules.

        This class allows to compute the i-RMSD, L-RMSD, Fnat and DockQ score of a given conformation.
        This can be a replacement for ProFIT. Note that the calculation of the zones are done by the class itself
        and does not require any extra input. This local class could be replaced by pdb2sql.StructureSimilarity.
        (https://github.com/DeepRank/pdb2sql)

        Args:
            decoy (str): file name of the decoy
            ref (str): file name of the reference
            verbose (bool, optional): print debug information


        Example :

        >>> from deeprank.tools import StructureSimilarity
        >>> decoy = '1AK4_1w.pdb'
        >>> ref = '1AK4.pdb'
        >>> sim = StructureSimilarity(decoy,ref)
        >>> irmsd_fast = sim.compute_irmsd_fast(method='svd',izone='1AK4.izone')
        >>> irmsd = sim.compute_irmsd_pdb2sql(method='svd',izone='1AK4.izone')
        >>> lrmsd_fast = sim.compute_lrmsd_fast(method='svd',lzone='1AK4.lzone',check=True)
        >>> lrmsd = sim.compute_lrmsd_pdb2sql(exportpath=None,method='svd')
        >>> Fnat = sim.compute_Fnat_pdb2sql()
        >>> Fnat_fast = sim.compute_Fnat_fast(ref_pairs='1AK4.ref_pairs')
        >>> dockQ = sim.compute_DockQScore(Fnat_fast,lrmsd_fast,irmsd_fast)

        """

        self.decoy = decoy
        self.ref = ref
        self.verbose = verbose

    def compute_lrmsd_fast(self,lzone=None,method='svd',check=True):
        """Fast routine to compute the L-RMSD.

        This routine parse the PDB directly without using pdb2sql.
        L-RMSD is computed by aligning the longest chain of the decoy to the one of the reference
        and computing the RMSD of the shortest chain between decoy and reference.

        Ref : DockQ: A Quality Measure for Protein-Protein Docking Models
              http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0161879

        Args:
            lzone (None, optional): name of the file containing the zone definition. If None the file will be calculated first
            method (str, optional): Method to align the fragments ('svd','quaternion')
            check (bool, optional): Check if the sequences are aligned and fix it if not. Should be True

        Returns:
            float: L-RMSD value of the conformation
        """

        # create/read the lzone file
        if lzone is None:
            resData = self.compute_lzone(save_file=False)
        elif not os.path.isfile(lzone):
            self.compute_lzone(save_file=True,filename=lzone)
            resData = self.read_zone(lzone)
        else:
            resData = self.read_zone(lzone)

        ##################################################
        # the check make sure that all the
        # atoms are in the correct order
        # I STRONGLY discourage turning the check off
        # it actually reorder the xyz data of the native/decoy
        # to match.
        ##################################################
        if check:

            data_decoy_long, data_decoy_short  = self.read_data_zone(self.decoy,resData,return_not_in_zone=True)
            data_ref_long,   data_ref_short    = self.read_data_zone(self.ref,resData,return_not_in_zone=True)

            atom_decoy_long = [ data[:3] for data in data_decoy_long ]
            atom_ref_long   = [ data[:3] for data in data_ref_long ]

            xyz_decoy_long, xyz_ref_long = [],[]
            for ind_decoy, at in enumerate(atom_decoy_long):

                try:
                    ind_ref = atom_ref_long.index(at)
                    xyz_decoy_long.append(data_decoy_long[ind_decoy][3:])
                    xyz_ref_long.append(data_ref_long[ind_ref][3:])
                except ValueError:
                    pass

            atom_decoy_short = [data[:3] for data in data_decoy_short]
            atom_ref_short   = [data[:3] for data in data_ref_short]

            xyz_decoy_short, xyz_ref_short = [],[]
            for ind_decoy, at in enumerate(atom_decoy_short):
                try:
                    ind_ref = atom_ref_short.index(at)
                    xyz_decoy_short.append(data_decoy_short[ind_decoy][3:])
                    xyz_ref_short.append(data_ref_short[ind_ref][3:])
                except ValueError:
                    pass


        # extract the xyz
        else:
            print('WARNING : The atom order have not been checked. Switch to check=True or continue at your own risk')
            xyz_decoy_long,xyz_decoy_short  = self.read_xyz_zone(self.decoy,resData,return_not_in_zone=True)
            xyz_ref_long,xyz_ref_short  = self.read_xyz_zone(self.ref,resData,return_not_in_zone=True)

        # get the translation so that both A chains are centered
        tr_decoy = self.get_trans_vect(xyz_decoy_long)
        tr_ref = self.get_trans_vect(xyz_ref_long)

        # translate everything for 1
        xyz_decoy_short = self.translation(xyz_decoy_short,tr_decoy)
        xyz_decoy_long = self.translation(xyz_decoy_long,tr_decoy)

        # translate everuthing for 2
        xyz_ref_short = self.translation(xyz_ref_short,tr_ref)
        xyz_ref_long = self.translation(xyz_ref_long,tr_ref)

        # get the ideql rotation matrix
        # to superimpose the A chains
        U = self.get_rotation_matrix(xyz_decoy_long,xyz_ref_long,method=method)

        # rotate the entire fragment
        xyz_decoy_short = self.rotation_matrix(xyz_decoy_short,U,center=False)

        # compute the RMSD
        return self.get_rmsd(xyz_decoy_short,xyz_ref_short)

    def compute_lzone(self,save_file=True,filename=None):
        """Compute the zone for L-RMSD calculation

        Args:
            save_file (bool, optional): save the zone file
            filename (str, optional): name of the file

        Returns:
            dict: definition of the zone
        """
        sql_ref = pdb2sql(self.ref)
        nA = len(sql_ref.get('x,y,z',chainID='A'))
        nB = len(sql_ref.get('x,y,z',chainID='B'))

        # detect which chain is the longest
        long_chain = 'A'
        if nA<nB:
            long_chain = 'B'

        # extract data about the residue
        data_test = [tuple(data) for data in sql_ref.get('chainID,resSeq',chainID=long_chain)]
        data_test = sorted(set(data_test))

        # close the sql
        sql_ref.close()

        if save_file:
            if filename is None:
                f = open(self.ref.split('.')[0]+'.lzone','w')
            else:
                f = open(filename,'w')
            for res in data_test:
                chain = res[0]
                num = res[1]
                f.write('zone %s%d-%s%d\n' %(chain,num,chain,num) )
            f.close()
            return

        else:
            resData = {}
            for res in data_test:
                chain = res[0]
                num = res[1]

                if chain not in resData.keys():
                    resData[chain] = []
                resData[chain].append(num)
            return resData

    def compute_irmsd_fast(self,izone=None,method='svd',cutoff=10.0,check=True):
        """Fast method to compute the i-rmsd

        Require the precalculation of the izone
        A dedicated routine is implemented to comoute the izone
        if izone is not given in argument the routine will compute them automatically

        i-RMSD is computed selecting the back bone of the contact residue with a cutoff of 10A
        in the decoy. Align these as best as possible with their coutner part in the ref
        and and compute the RMSD

        Ref : DockQ: A Quality Measure for Protein-Protein Docking Models
              http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0161879

        Args:
            izone (None, optional): file name of the zone. if None the zones will be calculated first
            method (str, optional): Method to align the fragments ('svd','quaternion')
            cutoff (float, optional): cutoff for the contact atoms
            check (bool, optional): Check if the sequences are aligned and fix it if not. Should be True

        Returns:
            float: i-RMSD value of the conformation
        """

        # read the izone file
        if izone is None:
            resData = self.compute_izone(cutoff,save_file=False)
        elif not os.path.isfile(izone):
            self.compute_izone(cutoff,save_file=True,filename=izone)
            resData = self.read_zone(izone)
        else:
            resData = self.read_zone(izone)

        ##################################################
        # the check make sure that all the
        # atoms are in the correct order
        # I STRONGLY discourage turning the check off
        # it actually fixes the order
        ##################################################
        if check:

            data_decoy  = self.read_data_zone(self.decoy,resData,return_not_in_zone=False)
            data_ref    = self.read_data_zone(self.ref,resData,return_not_in_zone=False)

            atom_decoy = [ data[:3] for data in data_decoy]
            atom_ref   = [ data[:3] for data in data_ref ]

            xyz_contact_decoy, xyz_contact_ref = [],[]
            for ind_decoy, at in enumerate(atom_decoy):
                try:
                    ind_ref = atom_ref.index(at)
                    xyz_contact_decoy.append(data_decoy[ind_decoy][3:])
                    xyz_contact_ref.append(data_ref[ind_ref][3:])
                except ValueError:
                    pass

        # extract the xyz
        else:
            print('WARNING : The atom order have not been checked. Switch to check=True or continue at your own risk')
            xyz_contact_decoy = self.read_xyz_zone(self.decoy,resData)
            xyz_contact_ref   = self.read_xyz_zone(self.ref,resData)

        # get the translation so that both A chains are centered
        tr_decoy = self.get_trans_vect(xyz_contact_decoy)
        tr_ref   = self.get_trans_vect(xyz_contact_ref)

        # translate everything
        xyz_contact_decoy = self.translation(xyz_contact_decoy,tr_decoy)
        xyz_contact_ref   = self.translation(xyz_contact_ref,tr_ref)

        # get the ideql rotation matrix
        # to superimpose the A chains
        U = self.get_rotation_matrix(xyz_contact_decoy,xyz_contact_ref,method=method)

        # rotate the entire fragment
        xyz_contact_decoy = self.rotation_matrix(xyz_contact_decoy,U,center=False)

        # return the RMSD
        return self.get_rmsd(xyz_contact_decoy,xyz_contact_ref)

    def compute_izone(self,cutoff=5.0,save_file=True,filename=None):
        """Compute the zones  for i-rmsd calculationss

        Args:
            cutoff (float, optional): cutoff for the contact atoms
            save_file (bool, optional): svae file containing the zone
            filename (str, optional): filename

        Returns:
            dict: i-zone definition
        """

        sql_ref = pdb2sql(self.ref)
        contact_ref = sql_ref.get_contact_atoms(cutoff=cutoff,extend_to_residue=True,return_only_backbone_atoms=True)
        index_contact_ref = contact_ref[0]+contact_ref[1]

        # get the xyz and atom identifier of the decoy contact atoms
        #xyz_contact_ref = sql_ref.get('x,y,z',rowID=index_contact_ref)
        data_test = [tuple(data) for data in sql_ref.get('chainID,resSeq',rowID=index_contact_ref)]
        data_test = sorted(set(data_test))

        # close the sql
        sql_ref.close()

        if save_file:

            if filename is None:
                f = open(self.ref.split('.')[0]+'.izone','w')
            else:
                f = open(filename,'w')

            for res in data_test:
                chain = res[0]
                num = res[1]
                f.write('zone %s%d-%s%d\n' %(chain,num,chain,num) )
            f.close()
            return

        else:
            resData = {}
            for res in data_test:
                chain = res[0]
                num = res[1]

                if chain not in resData.keys():
                    resData[chain] = []
                resData[chain].append(num)
            return resData

    def compute_Fnat_fast(self,ref_pairs=None,cutoff=5):
        """Compute the FNAT of the conformation

        Args:
            ref_pairs (str, optional): file name describing the pairs
            cutoff (int, optional): cutoff for the contact atoms

        Returns:
            float: FNAT value

        Raises:
            ValueError: if the decoy file is not found
        """
        # read the izone file
        if ref_pairs is None:
            residue_pairs_ref = self.compute_residue_pairs_ref(cutoff,save_file=False)

        elif not os.path.isfile(ref_pairs):
            self.compute_residue_pairs_ref(cutoff,save_file=True,filename=ref_pairs)
            f = open(ref_pairs,'rb')
            residue_pairs_ref = pickle.load(f)
            f.close()
        else:
            f = open(ref_pairs,'rb')
            residue_pairs_ref = pickle.load(f)
            f.close()

        # create a dict of the ecoy data
        if isinstance(self.decoy,str) and os.path.isfile(self.decoy):
            with open(self.decoy,'r') as f:
                data_decoy = f.readlines()
            decoy_name = os.path.basename(self.decoy)
        elif isinstance(self.decoy,np.ndarray):
            data_decoy = [l.decode('utf-8') for l in self.decoy]
            decoy_name = 'decoy'
        else:
            raise ValueError('Decoy not found')

        # read the decoy data
        atom_decoy, xyz_decoy = [],[]
        residue_xyz = {}
        residue_name = {}

        # go through all the lines
        # that starts with ATOM
        for line in data_decoy:

            if line.startswith('ATOM'):

                # chain ID
                chainID = line[21]
                if chainID == ' ':
                    chainID = line[72]

                # atom info
                resSeq = int(line[22:26])
                resName = line[17:20].strip()
                name = line[12:16].strip()

                # position
                x,y,z = float(line[30:38]), float(line[38:46]), float(line[46:54])

                # dict entry
                key = (chainID,resSeq,resName)

                # create the dict entry if necessary
                if key not in residue_xyz.keys():
                    residue_xyz[key] = []
                    residue_name[key] = []

                # we exclude the Hydrogens from the search
                if name[0] != 'H':
                    residue_xyz[key].append([x,y,z])
                    residue_name[key].append(name)

        # loop over the residue pairs of the
        # and increment common if an atom pair is close enough
        nCommon,nTotal = 0,0
        for resA,resB_list in residue_pairs_ref.items():
            if resA in residue_xyz:
                xyzA = residue_xyz[resA]
                for resB in resB_list:
                    if resB in residue_xyz.keys():
                        xyzB = residue_xyz[resB]
                        dist_min = np.min(np.array([  np.sqrt(np.sum((np.array(p1)-np.array(p2))**2)) for p1 in xyzA for p2 in xyzB  ]))
                        if dist_min<=cutoff:
                            nCommon += 1
                    nTotal += 1
            else:
                msg = '\t FNAT Warning could not find residue: ', resA, ' in: ',decoy_name
                _printif(msg,self.verbose)

        # normalize
        return nCommon/nTotal

    def compute_residue_pairs_ref(self,cutoff=5.0,save_file=True,filename=None):
        """Compute the residue pair on the reference conformation

        Args:
            cutoff (float, optional): cutoff for the contact atoms
            save_file (bool, optional): save the file containing the residue pairs
            filename (None, optional): filename

        Returns:
            dict: defintition of the residue pairs
        """
        sql_ref = pdb2sql(self.ref)
        residue_pairs_ref   = sql_ref.get_contact_residue(cutoff=cutoff,return_contact_pairs=True,
                                                          excludeH=True)
        sql_ref.close()

        if save_file:
            if filename is None:
                f = open(self.ref.split('.')[0]+'residue_contact_pairs.pckl','wb')
            else:
                f = open(filename,'wb')

            # save as pickle
            pickle.dump(residue_pairs_ref,f)
            f.close()
            return
        else:
            return residue_pairs_ref

    def compute_lrmsd_pdb2sql(self,exportpath=None,method='svd'):

        """Slow routine to compute the L-RMSD.

        This routine parse the PDB directly using pdb2sql.
        L-RMSD is computed by aligning the longest chain of the decoy to the one of the reference
        and computing the RMSD of the shortest chain between decoy and reference.

        Ref : DockQ: A Quality Measure for Protein-Protein Docking Models
              http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0161879

        Args:
            exportpath (str, optional): file name where the aligned pdbs are exported
            method (str, optional): Method to align the fragments ('svd','quaternion')

        Returns:
            float: L-RMSD value of the conformation
        """

        # create the sql
        sql_decoy = pdb2sql(self.decoy,sqlfile='decoy.db')
        sql_ref = pdb2sql(self.ref,sqlfile='ref.db')

        # extract the pos of chains A
        xyz_decoy_A = np.array(sql_decoy.get('x,y,z',chainID='A'))
        xyz_ref_A = np.array(sql_ref.get('x,y,z',chainID='A'))

        # extract the pos of chains B
        xyz_decoy_B = np.array(sql_decoy.get('x,y,z',chainID='B'))
        xyz_ref_B = np.array(sql_ref.get('x,y,z',chainID='B'))


        # check the lengthes
        if len(xyz_decoy_A) != len(xyz_ref_A):
            xyz_decoy_A, xyz_ref_A = self.get_identical_atoms(sql_decoy,sql_ref,'A')

        if len(xyz_decoy_B) != len(xyz_ref_B):
            xyz_decoy_B, xyz_ref_B = self.get_identical_atoms(sql_decoy,sql_ref,'B')


        # detect which chain is the longest
        nA,nB = len(xyz_decoy_A),len(xyz_decoy_B)
        if nA>nB:
            xyz_decoy_long = xyz_decoy_A
            xyz_ref_long = xyz_ref_A

            xyz_decoy_short = xyz_decoy_B
            xyz_ref_short = xyz_ref_B

        else:
            xyz_decoy_long = xyz_decoy_B
            xyz_ref_long = xyz_ref_B

            xyz_decoy_short = xyz_decoy_A
            xyz_ref_short = xyz_ref_A

        # get the translation so that both A chains are centered
        tr_decoy = self.get_trans_vect(xyz_decoy_long)
        tr_ref = self.get_trans_vect(xyz_ref_long)

        # translate everything for 1
        xyz_decoy_short = self.translation(xyz_decoy_short,tr_decoy)
        xyz_decoy_long = self.translation(xyz_decoy_long,tr_decoy)

        # translate everuthing for 2
        xyz_ref_short = self.translation(xyz_ref_short,tr_ref)
        xyz_ref_long = self.translation(xyz_ref_long,tr_ref)

        # get the ideal rotation matrix
        # to superimpose the A chains
        U = self.get_rotation_matrix(xyz_decoy_long,xyz_ref_long,method=method)

        # rotate the entire fragment
        xyz_decoy_short = self.rotation_matrix(xyz_decoy_short,U,center=False)


        # compute the RMSD
        lrmsd =  self.get_rmsd(xyz_decoy_short,xyz_ref_short)

        # export the pdb for verifiactions
        if exportpath is not None:

            # extract the pos of the dimer
            xyz_decoy = np.array(sql_decoy.get('x,y,z'))
            xyz_ref = np.array(sql_ref.get('x,y,z'))

            # translate
            xyz_ref = self.translation(xyz_ref,tr_ref)
            xyz_decoy = self.translation(xyz_decoy,tr_decoy)

            # rotate decoy
            xyz_decoy= self.rotation_matrix(xyz_decoy,U,center=False)

            # update the sql database
            sql_decoy.update_xyz(xyz_decoy)
            sql_ref.update_xyz(xyz_ref)

            # export
            sql_decoy.exportpdb(exportpath+'/lrmsd_decoy.pdb')
            sql_ref.exportpdb(exportpath+'/lrmsd_aligned.pdb')

        # close the db
        sql_decoy.close()
        sql_ref.close()

        return lrmsd

    @staticmethod
    def get_identical_atoms(db1,db2,chain):
        """Return that atoms shared by both databse for a specific chain

        Args:
            db1 (TYPE): pdb2sql database of the first conformation
            db2 (TYPE): pdb2sql database of the 2nd conformation
            chain (str): chain name

        Returns:
            list, list: list of xyz for both database
        """
        # get data
        data1 = db1.get('chainID,resSeq,name',chainID=chain)
        data2 = db2.get('chainID,resSeq,name',chainID=chain)

        # tuplify
        data1 = [tuple(d1) for d1 in data1]
        data2 = [tuple(d2) for d2 in data2]

        # get the intersection
        shared_data = list(set(data1).intersection(data2))

        # get the xyz
        xyz1,xyz2 = [],[]
        for data in shared_data:
            query = 'SELECT x,y,z from ATOM WHERE chainID=? AND resSeq=? and name=?'
            xyz1.append(list(list(db1.c.execute(query,data))[0]))
            xyz2.append(list(list(db2.c.execute(query,data))[0]))

        return xyz1,xyz2

    def compute_irmsd_pdb2sql(self,cutoff=10,method='svd',izone=None,exportpath=None):
        """Slow method to compute the i-rmsd

        Require the precalculation of the izone. A dedicated routine is implemented to comoute the izone
        if izone is not given in argument the routine will compute them automatically

        i-RMSD is computed selecting the back bone of the contact residue with a cutoff of 10A
        in the decoy. Align these as best as possible with their coutner part in the ref
        and and compute the RMSD

        Ref : DockQ: A Quality Measure for Protein-Protein Docking Models
              http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0161879

        Args:
            izone (None, optional): file name of the zone. if None the zones will be calculated first
            method (str, optional): Method to align the fragments ('svd','quaternion')
            cutoff (float, optional): cutoff for the contact atoms
            exportpath (str, optional): file name where the aligned pdbs are exported

        Returns:
            float: i-RMSD value of the conformation
        """

        # create thes sql
        sql_decoy = pdb2sql(self.decoy)
        sql_ref = pdb2sql(self.ref)

        # get the contact atoms
        if izone is None:
            contact_ref = sql_ref.get_contact_atoms(cutoff=cutoff,extend_to_residue=True,
                                                    return_only_backbone_atoms=False)
            index_contact_ref = contact_ref[0]+contact_ref[1]
        else:
            index_contact_ref = self.get_izone_rowID(sql_ref,izone,return_only_backbone_atoms=False)


        # get the xyz and atom identifier of the decoy contact atoms
        xyz_contact_ref = sql_ref.get('x,y,z',rowID=index_contact_ref)
        data_contact_ref = sql_ref.get('chainID,resSeq,resName,name',rowID=index_contact_ref)

        # get the xyz and atom indeitifier of the reference
        xyz_decoy = sql_decoy.get('x,y,z')
        data_decoy = sql_decoy.get('chainID,resSeq,resName,name')

        # loop through the ref label
        # check if the atom is in the decoy
        # if yes -> add xyz to xyz_contact_decoy
        # if no  -> remove the corresponding to xyz_contact_ref
        xyz_contact_decoy = []
        index_contact_decoy = []
        clean_ref = False
        for iat,atom in enumerate(data_contact_ref):

            try:
                index = data_decoy.index(atom)
                index_contact_decoy.append(index)
                xyz_contact_decoy.append(xyz_decoy[index])
            except Exception:
                xyz_contact_ref[iat] = None
                index_contact_ref[iat] = None
                clean_ref = True

        # clean the xyz
        if clean_ref:
            xyz_contact_ref = [xyz for xyz in xyz_contact_ref if xyz is not None]
            index_contact_ref = [ind for ind in index_contact_ref if ind is not None]


        # check that we still have atoms in both chains
        chain_decoy = list(set(sql_decoy.get('chainID',rowID=index_contact_decoy)))
        chain_ref   = list(set(sql_ref.get('chainID',rowID=index_contact_ref)))

        if len(chain_decoy)<1 or len(chain_ref)<1:
            raise ValueError('Error in i-rmsd: only one chain represented in one chain')


        # get the translation so that both A chains are centered
        tr_decoy = self.get_trans_vect(xyz_contact_decoy)
        tr_ref   = self.get_trans_vect(xyz_contact_ref)

        # translate everything
        xyz_contact_decoy = self.translation(xyz_contact_decoy,tr_decoy)
        xyz_contact_ref   = self.translation(xyz_contact_ref,tr_ref)

        # get the ideql rotation matrix
        # to superimpose the A chains
        U = self.get_rotation_matrix(xyz_contact_decoy,xyz_contact_ref,method=method)

        # rotate the entire fragment
        xyz_contact_decoy = self.rotation_matrix(xyz_contact_decoy,U,center=False)

        # compute the RMSD
        irmsd = self.get_rmsd(xyz_contact_decoy,xyz_contact_ref)



        # export the pdb for verifiactions
        if exportpath is not None:

            # update the sql database
            sql_decoy.update_xyz(xyz_contact_decoy,index=index_contact_decoy)
            sql_ref.update_xyz(xyz_contact_ref,index=index_contact_ref)

            sql_decoy.exportpdb(exportpath+'/irmsd_decoy.pdb',index=index_contact_decoy)
            sql_ref.exportpdb(exportpath+'/irmsd_ref.pdb',index=index_contact_ref)

        # close the db
        sql_decoy.close()
        sql_ref.close()

        return irmsd

    @staticmethod
    def get_izone_rowID(sql,izone,return_only_backbone_atoms=True):
        """Compute the index of the izone atoms

        Args:
            sql (pdb2sql): database of the conformation
            izone (str): filename to store the zone
            return_only_backbone_atoms (bool, optional): Returns only the backbone atoms

        Returns:
            lis(int): index of the atoms in the zone

        Raises:
            FileNotFoundError: if the izone file is not found
        """
        # read the file
        if not os.path.isfile(izone):
            raise FileNotFoundError('i-zone file not found',izone)

        with open(izone,'r') as f:
            data=f.readlines()

        # get the data out of it
        resData = {}
        for line in data:

            res = line.split()[1].split('-')[0]
            chainID,resSeq = res[0],int(res[1:])


            if chainID not in resData.keys():
                resData[chainID] = []

            resData[chainID].append(resSeq)

        # get the rowID
        index_contact = []

        for chainID,resSeq in resData.items():
            if return_only_backbone_atoms:
                index_contact += sql.get('rowID',chainID=chainID,resSeq=resSeq,
                                          name=['C','CA','N','O'])
            else:
                index_contact += sql.get('rowID',chainID=chainID,resSeq=resSeq)

        return index_contact

    def compute_Fnat_pdb2sql(self,cutoff=5.0):
        """Slow method to compute the FNAT usign pdb2sql

        Args:
            cutoff (float, optional): cutoff for the contact atoms

        Returns:
            float: Fnat value for the conformation
        """
        # create the sql
        sql_decoy = pdb2sql(self.decoy)
        sql_ref = pdb2sql(self.ref)

        # get the contact atoms of the decoy
        residue_pairs_decoy = sql_decoy.get_contact_residue(cutoff=cutoff,
                                                            return_contact_pairs=True,
                                                            excludeH=True)

        # get the contact atoms of the ref
        residue_pairs_ref   = sql_ref.get_contact_residue(cutoff=cutoff,
                                                          return_contact_pairs=True,
                                                          excludeH=True)


        # form the pair data
        data_pair_decoy = []
        for resA,resB_list in residue_pairs_decoy.items():
            data_pair_decoy += [  (resA,resB) for resB in resB_list   ]

        # form the pair data
        data_pair_ref = []
        for resA,resB_list in residue_pairs_ref.items():
            data_pair_ref += [  (resA,resB) for resB in resB_list     ]


        # find the umber of residue that ref and decoys hace in common
        nCommon = len(set(data_pair_ref).intersection(data_pair_decoy))

        # normalize
        Fnat = nCommon/len(data_pair_ref)

        sql_decoy.close()
        sql_ref.close()

        return Fnat



    ################################################################################################
    #
    #   HELPER ROUTINES TO HANDLE THE ZONE FILES
    #
    #################################################################################################


    @staticmethod
    def read_xyz_zone(pdb_file,resData,return_not_in_zone=False):
        """Read the xyz of the zone atoms

        Args:
            pdb_file (str): filename containing the pdb of the molecule
            resData (dict): information about the residues
            return_not_in_zone (bool, optional): Do we return the atoms not in the zone

        Returns:
            list(float): XYZ of the atoms in the zone
        """
        # read the ref file
        with open(pdb_file,'r') as f:
            data = f.readlines()

        # get the xyz of the
        xyz_in_zone = []
        xyz_not_in_zone = []

        for line in data:

            if line.startswith('ATOM'):

                chainID = line[21]
                if chainID == ' ':
                    chainID = line[72]

                resSeq = int(line[22:26])
                name = line[12:16].strip()

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                if chainID in resData.keys():

                    if resSeq in resData[chainID] and name in ['C','CA','N','O']:
                        xyz_in_zone.append([x,y,z])

                    elif resSeq not in resData[chainID] and name in ['C','CA','N','O']:
                        xyz_not_in_zone.append([x,y,z])

                else:
                    if name in ['C','CA','N','O']:
                        xyz_not_in_zone.append([x,y,z])

        if return_not_in_zone:
            return xyz_in_zone,xyz_not_in_zone

        else:
            return xyz_in_zone


    @staticmethod
    def read_data_zone(pdb_file,resData,return_not_in_zone=False):
        """Read the data of the atoms in the zone.

        Args:
            pdb_file (str): filename containing the pdb of the molecule
            resData (dict): information about the residues
            return_not_in_zone (bool, optional): Do we return the atoms not in the zone

        Returns:
            list(float): data of the atoms in the zone
        """
        # read the ref file
        if isinstance(pdb_file,str) and os.path.isfile(pdb_file):
            with open(pdb_file,'r') as f:
                data = f.readlines()
        elif isinstance(pdb_file,np.ndarray):
            data =  [l.decode('utf-8') for l in pdb_file]

        # get the xyz of the
        data_in_zone = []
        data_not_in_zone = []

        for line in data:

            if line.startswith('ATOM'):

                chainID = line[21]
                if chainID == ' ':
                    chainID = line[72]

                resSeq = int(line[22:26])
                name = line[12:16].strip()

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                if chainID in resData.keys():

                    if resSeq in resData[chainID] and name in ['C','CA','N','O']:
                        data_in_zone.append([chainID,resSeq,name,x,y,z])

                    elif resSeq not in resData[chainID] and name in ['C','CA','N','O']:
                        data_not_in_zone.append([chainID,resSeq,name,x,y,z])

                else:
                    if name in ['C','CA','N','O']:
                        data_not_in_zone.append([chainID,resSeq,name,x,y,z])

        if return_not_in_zone:
            return data_in_zone,data_not_in_zone

        else:
            return data_in_zone


    @staticmethod
    def read_zone(zone_file):
        """Read the zone file.

        Args:
            zone_file (str): name of the file

        Returns:
            dict: Info aboyt the residues in the zone

        Raises:
            FileNotFoundError: if the zone file is not found
        """
        # read the izone file
        if not os.path.isfile(zone_file):
            raise FileNotFoundError('zone file not found',zone_file)

        with open(zone_file,'r') as f:
            data=f.readlines()

        # get the data out of it
        resData = {}
        for line in data:

            #    line = zone A4-A4   for positive resNum
            # or line = zone A-4-A-4 for negative resNum
            # that happens for example in 2OUL

            # split the line
            res = line.split()[1].split('-')

            # if the resnum was positive
            # we have e.g res = [A4,A4]
            if len(res) == 2:
                res = res[0]
                chainID,resSeq = res[0],int(res[1:])

            # if the resnum was negative was negtive
            # we have e.g res = [A,4,A,4]
            elif len(res) == 4:
                chainID,resSeq = res[0],-int(res[1])

            if chainID not in resData.keys():
                resData[chainID] = []

            resData[chainID].append(resSeq)

        return resData


    ###################################################################
    #
    #   ROUTINES TO ACTUALY ALIGN THE MOLECULES
    #
    ###################################################################

    @staticmethod
    def compute_DockQScore(Fnat,lrmsd,irmsd,d1=8.5,d2=1.5):
        """Compute the DockQ Score

        Args:
            Fnat (float): Fnat value
            lrmsd (float): lrmsd value
            irmsd (float): irmsd value
            d1 (float, optional): first coefficient for the DockQ calculations
            d2 (float, optional): second coefficient for the DockQ calculations
        """
        def scale_rms(rms,d):
            return(1./(1+(rms/d)**2))

        return 1./3 * (  Fnat + scale_rms(lrmsd,d1) + scale_rms(irmsd,d2))

    @staticmethod
    def get_rmsd(P,Q):
        """compute the RMSD

        Args:
            P (np.array(nx3)): position of the points in the first molecule
            Q (np.array(nx3)): position of the points in the second molecule

        Returns:
            float: RMSD value
        """
        n = len(P)
        return np.sqrt(1./n*np.sum((P-Q)**2))

    @staticmethod
    def get_trans_vect(P):
        """Get the translationv vector to the origin

        Args:
            P (np.array(nx3)): position of the points in the molecule

        Returns:
            float: minus mean value of the xyz columns
        """
        return  -np.mean(P,0)


    # main switch for the rotation matrix
    # add new methods here if necessary
    def get_rotation_matrix(self,P,Q,method='svd'):

        # get the matrix with Kabsh method
        if method.lower()=='svd':
            return self.get_rotation_matrix_Kabsh(P,Q)

        # or with the quaternion method
        elif method.lower()=='quaternion':
            return self.get_rotation_matrix_quaternion(P,Q)

        else:
            raise ValueError('%s is not a valid method for rmsd alignement.\n Options are svd or quaternions' %method)

    @staticmethod
    def get_rotation_matrix_Kabsh(P,Q):
        '''Get the rotation matrix to aligh two point clouds.

        The method is based on th Kabsh approach
        https://cnx.org/contents/HV-RsdwL@23/Molecular-Distance-Measures

        Args:
            P (np.array): xyz of the first point cloud
            Q (np.array): xyz of the second point cloud

        Returns:
            np.array: rotation matrix

        Raises:
            ValueError: matrix have different sizes
        '''

        pshape = P.shape
        qshape = Q.shape

        if pshape[0] == qshape[0]:
            npts = pshape[0]
        else:
            raise ValueError("Matrix don't have the same number of points",P.shape,Q.shape)


        p0,q0 = np.abs(np.mean(P,0)),np.abs(np.mean(Q,0))
        eps = 1E-6
        if any(p0 > eps) or any(q0 > eps):
            raise ValueError('You must center the fragment first',p0,q0)


        # form the covariance matrix
        A = np.dot(P.T,Q)/npts

        # SVD the matrix
        V,S,W = np.linalg.svd(A)

        # the W matrix returned here is
        # already its transpose
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.svd.html
        W = W.T

        # determinant
        d = np.linalg.det(np.dot(W,V.T))

        # form the U matrix
        Id = np.eye(3)
        if d < 0:
            Id[2,2] = -1

        U = np.dot(W,np.dot(Id,V.T))

        return U

    @staticmethod
    def get_rotation_matrix_quaternion(P,Q):
        '''Get the rotation matrix to aligh two point clouds

        The method is based on the quaternion approach
        http://www.ams.stonybrook.edu/~coutsias/papers/rmsd17.pdf

        Args:
            P (np.array): xyz of the first point cloud
            Q (np.array): xyz of the second point cloud

        Returns:
            np.array: rotation matrix

        Raises:
            ValueError: matrix have different sizes
        '''

        pshape = P.shape
        qshape = Q.shape

        if pshape[0] != qshape[0]:
            raise ValueError("Matrix don't have the same number of points",P.shape,Q.shape)

        p0,q0 = np.abs(np.mean(P,0)),np.abs(np.mean(Q,0))
        eps = 1E-6
        if any(p0 > eps) or any(q0 > eps):
            raise ValueError('You must center the fragment first',p0,q0)

        # form the correlation matrix
        R = np.dot(P.T,Q)

        # form the F matrix (eq. 10 of ref[1])
        F = np.zeros((4,4))

        F[0,0] = np.trace(R)
        F[0,1] = R[1,2]-R[2,1]
        F[0,2] = R[2,0]-R[0,2]
        F[0,3] = R[0,1]-R[1,0]

        F[1,0] = R[1,2]-R[2,1]
        F[1,1] = R[0,0]-R[1,1]-R[2,2]
        F[1,2] = R[0,1]+R[1,0]
        F[1,3] = R[0,2]+R[2,0]

        F[2,0] = R[2,0]-R[0,2]
        F[2,1] = R[0,1]+R[1,0]
        F[2,2] =-R[0,0]+R[1,1]-R[2,2]
        F[2,3] = R[1,2]+R[2,1]

        F[3,0] = R[0,1]-R[1,0]
        F[3,1] = R[0,2]+R[2,0]
        F[3,2] = R[1,2]+R[2,1]
        F[3,3] =-R[0,0]-R[1,1]+R[2,2]

        # diagonalize it
        l,U = np.linalg.eig(F)

        # extract the eigenvect of the highest eigenvalues
        indmax = np.argmax(l)
        q0,q1,q2,q3 = U[:,indmax]

        # form the rotation matrix (eq. 33 ref[1])
        U = np.zeros((3,3))

        U[0,0] = q0**2+q1**2-q2**2-q3**2
        U[0,1] = 2*(q1*q2-q0*q3)
        U[0,2] = 2*(q1*q3+q0*q2)
        U[1,1] = 2*(q1*q2+q0*q3)
        U[1,2] = q0**2-q1**2+q2*2-q3**2
        U[1,2] = 2*(q2*q3-q0*q1)
        U[2,0] = 2*(q1*q3-q0*q2)
        U[2,1] = 2*(q2*q3+q0*q1)
        U[2,2] = q0**2-q1**2-q2**2+q3**2

        return U


    @staticmethod
    def translation(xyz,vect):
        """Translate a fragment

        Args:
            xyz (np.array): position of the fragment
            vect (np.array): translation vector

        Returns:
            np.array: translated positions
        """
        return xyz + vect

    @staticmethod
    def rotation_around_axis(xyz,axis,angle):
        """Rotate a fragment around an axis.

        Args:
            xyz (np.array): original positions
            axis (np.array): axis of rotation
            angle (float): angle of rotation (radians)

        Returns:
            np.array: Rotated positions
        """

        # get the data
        ct,st = np.cos(angle),np.sin(angle)
        ux,uy,uz = axis

        # get the center of the molecule
        xyz0 = np.mean(xyz,0)

        # definition of the rotation matrix
        # see https://en.wikipedia.org/wiki/Rotation_matrix
        rot_mat = np.array([
        [ct + ux**2*(1-ct),         ux*uy*(1-ct) - uz*st,       ux*uz*(1-ct) + uy*st],
        [uy*ux*(1-ct) + uz*st,      ct + uy**2*(1-ct),          uy*uz*(1-ct) - ux*st],
        [uz*ux*(1-ct) - uy*st,      uz*uy*(1-ct) + ux*st,       ct + uz**2*(1-ct)   ]])

        # apply the rotation
        return np.dot(rot_mat,(xyz-xyz0).T).T + xyz0

    @staticmethod
    def rotation_euler(xyz,alpha,beta,gamma):
        """Rotate a fragment from Euler rotation angle

        Args:
            xyz (np.array): original positions
            alpha (float): rotation angle around the x axis
            beta (float): rotation angle around the x axis
            gamma (float): rotation angle around the x axis

        Returns:
            np.array: Rotated positions
        """
        # precompute the trig
        ca,sa = np.cos(alpha),np.sin(alpha)
        cb,sb = np.cos(beta),np.sin(beta)
        cg,sg = np.cos(gamma),np.sin(gamma)


        # get the center of the molecule
        xyz0 = np.mean(xyz,0)

        # rotation matrices
        rx = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
        ry = np.array([[cb,0,sb],[0,1,0],[-sb,0,cb]])
        rz = np.array([[cg,-sg,0],[sg,cs,0],[0,0,1]])

        rot_mat = np.dot(rx,np.dot(ry,rz))

        # apply the rotation
        return np.dot(rot_mat,(xyz-xyz0).T).T + xyz0


    @staticmethod
    def rotation_matrix(xyz,rot_mat,center=True):
        """Rotate a fragment from a roation matrix

        Args:
            xyz (np.array): original positions
            rot_mat (np.array): rotation matrix
            center (bool, optional): Center the fragment before rotation

        Returns:
            np.array: rotated positions
        """
        if center:
            xyz0 = np.mean(xyz)
            return np.dot(rot_mat,(xyz-xyz0).T).T + xyz0
        else:
            return np.dot(rot_mat,(xyz).T).T



# if __name__ == '__main__':
#     import time
#     BM4 = '/home/nico/Documents/projects/deeprank/data/HADDOCK/BM4_dimers/'
#     decoy = BM4 + 'decoys_pdbFLs/1AK4/water/1AK4_1w.pdb'
#     ref = BM4 + 'BM4_dimers_bound/pdbFLs_ori/1AK4.pdb'

#     sim = StructureSimilarity(decoy,ref)

#     #----------------------------------------------------------------------

#     t0 = time.time()
#     irmsd_fast = sim.compute_irmsd_fast(method='svd',izone='1AK4.izone')
#     t1 = time.time()-t0
#     print('\nIRMSD TIME FAST %f in %f sec' %(irmsd_fast,t1))

#     t0 = time.time()
#     irmsd = sim.compute_irmsd_pdb2sql(method='svd',izone='1AK4.izone')
#     t1 = time.time()-t0
#     print('IRMSD TIME SQL %f in %f sec' %(irmsd,t1))

#     #----------------------------------------------------------------------

#     t0 = time.time()
#     lrmsd_fast = sim.compute_lrmsd_fast(method='svd',lzone='1AK4.lzone',check=True)
#     t1 = time.time()-t0
#     print('\nLRMSD TIME FAST %f in %f sec' %(lrmsd_fast,t1))

#     t0 = time.time()
#     lrmsd = sim.compute_lrmsd_pdb2sql(exportpath=None,method='svd')
#     t1 = time.time()-t0
#     print('LRMSD TIME SQL %f in %f sec' %(lrmsd,t1))

#     #----------------------------------------------------------------------

#     t0 = time.time()
#     Fnat = sim.compute_Fnat_pdb2sql()
#     t1 = time.time()-t0
#     print('\nFNAT TIME SQL %f in %f sec' %(Fnat,t1))


#     t0 = time.time()
#     Fnat_fast = sim.compute_Fnat_fast(ref_pairs='1AK4.ref_pairs')
#     t1 = time.time()-t0
#     print('LRMSD TIME FAST %f in %f sec' %(Fnat_fast,t1))

#     #----------------------------------------------------------------------

#     dockQ = sim.compute_DockQScore(Fnat_fast,lrmsd_fast,irmsd_fast)
#     print('\nDockQ  %f' %dockQ )
