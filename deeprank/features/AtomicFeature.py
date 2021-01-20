import os
import warnings

import numpy as np
import pdb2sql

from deeprank.features import FeatureClass


class AtomicFeature(FeatureClass):

    def __init__(self, pdbfile, chain1='A', chain2='B', param_charge=None,
                param_vdw=None, patch_file=None, contact_cutoff=8.5,
                verbose=False):
        """Compute the Coulomb, van der Waals interaction and charges.

        Args:

            pdbfile (str): pdb file of the molecule

            chain1 (str): First chain ID, defaults to 'A'
            chain2 (str): Second chain ID, defaults to 'B'

            param_charge (str): file name of the force field file
                containing the charges e.g. protein-allhdg5.4_new.top.
                Must be of the format:
                * CYM  atom O   type=O      charge=-0.500 end
                * ALA    atom N   type=NH1     charge=-0.570 end

            param_vdw (str): file name of the force field containing
                vdw parameters e.g. protein-allhdg5.4_new.param.
                Must be of the format:
                * NONBonded  CYAA    0.105   3.750       0.013    3.750
                * NONBonded  CCIS    0.105   3.750       0.013    3.750

            patch_file (str): file name of a valid patch file for
                the parameters e.g. patch.top.
                The way we handle the patching is very manual and
                should be made more automatic.

            contact_cutoff (float): the maximum distance in Å
                between 2 contact atoms.

            verbose (bool): print or not.

        Examples:
            >>> pdb = '1AK4_100w.pdb'
            >>>
            >>> # get the force field included in deeprank
            >>> # if another FF has been used to compute the ref
            >>> # change also this path to the correct one
            >>> FF = pkg_resources.resource_filename(
            >>>     'deeprank.features','') + '/forcefield/'
            >>>
            >>> # declare the feature calculator instance
            >>> atfeat = AtomicFeature(pdb,
            >>>    param_charge = FF + 'protein-allhdg5-4_new.top',
            >>>    param_vdw    = FF + 'protein-allhdg5-4_new.param',
            >>>    patch_file   = FF + 'patch.top')
            >>>
            >>> # assign parameters
            >>> atfeat.assign_parameters()
            >>>
            >>> # only compute the pair interactions here
            >>> atfeat.evaluate_pair_interaction(save_interactions=test_name)
            >>>
            >>> # close the db
            >>> atfeat.sqldb._close()
        """

        super().__init__("Atomic")

        # set a few things
        self.pdbfile = pdbfile
        self.chain1 = chain1
        self.chain2 = chain2
        self.param_charge = param_charge
        self.param_vdw = param_vdw
        self.patch_file = patch_file
        self.contact_cutoff = contact_cutoff
        self.verbose = verbose

        # a few constant
        self.eps0 = 1
        self.c = 332.0636
        self.residue_key = 'chainID, resSeq, resName'
        self.atom_key = 'chainID, resSeq, resName, name'

        # read the pdb as an sql
        self.sqldb = pdb2sql.pdb2sql(self.pdbfile)

        # read the force field
        self.read_charge_file()

        if patch_file is not None:
            self.read_patch()
        else:
            self.patch_charge, self.patch_type = {}, {}

        # read the vdw param file
        self.read_vdw_file()

        # get the contact atoms
        self.get_contact_atoms()

    ####################################################################
    #
    #   READ INPUT FILES
    #
    ####################################################################

    def read_charge_file(self):
        """Read the .top file given in entry.

        This function creates:

        - self.charge: dictionary  {(resname,atname):charge}
        - self.valid_resnames: list ['VAL','ALP', .....]
        - self.at_name_type_convertor: dict {(resname,atname):attype}
        """

        with open(self.param_charge) as f:
            data = f.readlines()

        # loop over all the data
        self.charge = {}
        self.at_name_type_convertor = {}
        resnames = []

        # loop over the file
        for l in data:

            # split the line
            words = l.split()

            # get the resname/atname
            res, atname = words[0], words[2]

            # get the charge
            ind = l.find('charge=')
            q = float(l[ind + 7:ind + 13])

            # get the type
            attype = words[3].split('=')[-1]

            # store the charge
            self.charge[(res, atname)] = q

            # put the resname in a list so far
            resnames.append(res)

            # dictionary for conversion name/type
            self.at_name_type_convertor[(res, atname)] = attype

        self.valid_resnames = list(set(resnames))

    def read_patch(self):
        """Read the patchfile.

        This function creates

            - self.patch_charge: Dict {(resName,atName): charge}
            - self.patch_type : Dict {(resName,atName): type}
        """

        with open(self.patch_file) as f:
            data = f.readlines()

        self.patch_charge, self.patch_type = {}, {}

        for l in data:
            # ignore comments
            if l[0] != '#' and l[0] != '!' and len(l.split()) > 0:
                words = l.split()

                # get the new charge
                ind = l.find('CHARGE=')
                q = float(l[ind + 7:ind + 13])
                self.patch_charge[(words[0], words[3])] = q

                # get the new type if any
                ind = l.find('TYPE=')
                if ind != -1:
                    type_ = l[ind + 5:ind + 9]
                    self.patch_type[(words[0], words[3])] = type_.strip()

    def read_vdw_file(self):
        """Read the .param file.

        The param file must be of the form:

            NONBONDED ATNAME 0.10000 3.298765 0.100000 3.089222

            - First two numbers are for inter-chain interations
            - Last two nmbers are for intra-chain interactions
              (We only compute the interchain here)

        This function creates

            - self.vdw: dictionary {attype:[E1,S1]}
        """

        with open(self.param_vdw) as f:
            data = f.readlines()

        self.vdw_param = {}

        for line in data:
            # split the atom
            line = line.split()

            # empty line
            if len(line) == 0:
                continue

            # comment
            if line[0][0] == '#':
                continue

            self.vdw_param[line[1]] = list(map(float, line[2:4]))

    def get_contact_atoms(self):
        """Get the contact atoms only select amino acids.

        The ligands are not considered.
        """
        # TODO: replace this function with pdb2sql.get_contact_atoms
        # but need to add a filter parameter to filter out ligand.

        # position of the chains
        xyz1 = np.array(self.sqldb.get('x,y,z', chainID=self.chain1))
        xyz2 = np.array(self.sqldb.get('x,y,z', chainID=self.chain2))

        # rowID of the chains
        index_a = self.sqldb.get('rowID', chainID=self.chain1)
        index_b = self.sqldb.get('rowID', chainID=self.chain2)

        # resName of the chains
        resName1 = np.array(self.sqldb.get('resName', chainID=self.chain1))
        resName2 = np.array(self.sqldb.get('resName', chainID=self.chain2))

        # declare the contact atoms
        self.contact_atoms_A = []
        self.contact_atoms_B = []

        # The contact atom pairs only contains pairs of atoms that are
        # in contact
        self.contact_pairs = {}

        for i, x0 in enumerate(xyz1):

            # compute the contact atoms
            contacts = np.where(
                np.sqrt(np.sum((xyz2 - x0)**2, 1)) < self.contact_cutoff)[0]

            # if we have contact atoms and resA is not a ligand
            if (len(contacts) > 0) and (resName1[i] in self.valid_resnames):

                # add i to the list
                # add the index of b if its resname is not a ligand
                self.contact_atoms_A += [index_a[i]]
                self.contact_atoms_B += [
                    index_b[k] for k in contacts
                    if resName2[k] in self.valid_resnames
                ]

                # add the contact pairs to the list
                self.contact_pairs[index_a[i]] = [
                    index_b[k] for k in contacts
                    if resName2[k] in self.valid_resnames
                ]

        # create a set of unique indexes
        self.contact_atoms_A = sorted(set(self.contact_atoms_A))
        self.contact_atoms_B = sorted(set(self.contact_atoms_B))

        # if no interface atoms were found
        if len(self.contact_atoms_A) == 0:
            raise ValueError("No contact atoms detected in AtomicFeature")

    def _extend_contact_to_residue(self):
        """Extend the contact atoms to entire residue where one atom is
        contacting."""

        # extract the data
        dataA = self.sqldb.get(self.residue_key,
                               rowID=self.contact_atoms_A)
        dataB = self.sqldb.get(self.residue_key,
                               rowID=self.contact_atoms_B)

        # create tuple cause we want to hash through it
        dataA = [tuple(x) for x in dataA]
        dataB = [tuple(x) for x in dataB]

        # extract uniques
        resA = list(set(dataA))
        resB = list(set(dataB))

        # init the list
        index_contact_A, index_contact_B = [], []

        # contact of chain A
        for resdata in resA:
            chainID, resSeq, resName = resdata
            index_contact_A += self.sqldb.get('rowID', chainID=chainID,
                                              resName=resName, resSeq=resSeq)

        # contact of chain B
        for resdata in resB:
            chainID, resSeq, resName = resdata
            index_contact_B += self.sqldb.get('rowID', chainID=chainID,
                                              resName=resName, resSeq=resSeq)

        # make sure that we don't have double (maybe optional)
        index_contact_A = sorted(set(index_contact_A))
        index_contact_B = sorted(set(index_contact_B))

        return index_contact_A, index_contact_B

    ####################################################################
    #
    #   Assign parameters
    #
    ####################################################################

    def assign_parameters(self):
        """Assign to each atom in the pdb its charge and vdw interchain
        parameters.

        Directly deals with the patch so that we don't loop over the
        residues multiple times.
        """

        # get all the resnumbers
        if self.verbose:
            print('-- Assign force field parameters')

        data = self.sqldb.get(self.residue_key)
        natom = len(data)
        data = np.unique(np.array(data), axis=0)

        # declare the parameters for future insertion in SQL
        atcharge = np.zeros(natom)
        ateps = np.zeros(natom)
        atsig = np.zeros(natom)

        # check
        attype = np.zeros(natom, dtype='<U5')
        ataltResName = np.zeros(natom, dtype='<U5')

        # loop over all the residues
        for chain, resNum, resName in data:

            # atom types of the residue
            #query = "WHERE chainID='%s' AND resSeq=%s" %(chain,resNum)
            atNames = np.array(self.sqldb.get(
                'name', chainID=chain, resSeq=resNum))
            rowID = np.array(self.sqldb.get(
                'rowID', chainID=chain, resSeq=resNum))

            # get the alternative resname
            altResName = self._get_altResName(resName, atNames)

            # get the charge of this residue
            atcharge[rowID] = self._get_charge(resName, altResName, atNames)

            # get the vdw parameters
            eps, sigma, type_ = self._get_vdw(resName, altResName, atNames)
            ateps[rowID] += eps
            atsig[rowID] += sigma

            ataltResName[rowID] = altResName
            attype[rowID] = type_

        # put the charge in SQL
        self.sqldb.add_column('CHARGE')
        self.sqldb.update_column('CHARGE', atcharge)

        # put the VDW in SQL
        self.sqldb.add_column('eps')
        self.sqldb.update_column('eps', ateps)

        self.sqldb.add_column('sig')
        self.sqldb.update_column('sig', atsig)

        self.sqldb.add_column('type', 'TEXT')
        self.sqldb.update_column('type', attype)

        self.sqldb.add_column('altRes', 'TEXT')
        self.sqldb.update_column('altRes', ataltResName)

    @staticmethod
    def _get_altResName(resName, atNames):
        """Apply the patch data.

        This is adopted from preScan.pl
        This is very static and I don't quite like it
        The structure of the dictionary is as following

        { NEWRESTYPE: 'OLDRESTYPE',
                       [atom types that must be present],
                       [atom types that must NOT be present]]}

        Args:
            resName (str): name of the residue
            atNames (list(str)): names of the atoms
        """

        new_type = {
            'PROP': ['all', ['HT1', 'HT2'], []],
            'NTER': ['all', ['HT1', 'HT2', 'HT3'], []],
            'CTER': ['all', ['OXT'], []],
            'CTN': ['all', ['NT', 'HT1', 'HT2'], []],
            'CYNH': ['CYS', ['1SG'], ['2SG']],
            'DISU': ['CYS', ['1SG', '2SG'], []],
            'HISE': ['HIS', ['ND1', 'CE1', 'CD2', 'NE2', 'HE2'], ['HD1']],
            'HISD': ['HIS', ['ND1', 'CE1', 'CD2', 'NE2', 'HD1'], ['HE2']]
        }

        # this works fine now

        altResName = resName
        for key, values in new_type.items():
            res, atpres, atabs = values
            if res == resName or res == 'all':
                if all(x in atNames for x in atpres) and all(
                        x not in atNames for x in atabs):
                    altResName = key

        return altResName

    def _get_vdw(self, resName, altResName, atNames):
        """Get vdw itneraction terms.

        Args:
            resName (str): name of the residue
            altResName (str): alternative name of the residue
            atNames (list(str)): names of the atoms
        """

        # in case the resname is not valid
        if resName not in self.valid_resnames:
            vdw_eps = [0.00] * len(atNames)
            vdw_sigma = [0.00] * len(atNames)
            type_ = ['None'] * len(atNames)

            return vdw_eps, vdw_sigma, type_

        vdw_eps, vdw_sigma, type_ = [], [], []

        for at in atNames:

            if (altResName, at) in self.patch_type:
                type_.append(self.patch_type[(altResName, at)])
                vdw_data = self.vdw_param[self.patch_type[(altResName, at)]]
                vdw_eps.append(vdw_data[0])
                vdw_sigma.append(vdw_data[1])

            elif (resName, at) in self.at_name_type_convertor:
                type_.append(self.at_name_type_convertor[(resName, at)])
                vdw_data = self.vdw_param[self.at_name_type_convertor[(
                    resName, at)]]
                vdw_eps.append(vdw_data[0])
                vdw_sigma.append(vdw_data[1])

            else:
                type_.append('None')
                vdw_eps.append(0.0)
                vdw_sigma.append(0.0)
                warnings.warn(f"Atom type {at} not found for "
                              f"resType {resName} or patch type {altResName}. "
                              f"Set vdw eps and sigma to 0.0.")

        return vdw_eps, vdw_sigma, type_

    def _get_charge(self, resName, altResName, atNames):
        """Get the charge information.

        Args:
            resName (str): name of the residue
            altResName (str): alternative name of the residue
            atNames (list(str)): names of the atoms
        """
        # in case the resname is not valid
        if resName not in self.valid_resnames:
            q = [0.0] * len(atNames)
            return q

        # assign the charges
        q = []
        for at in atNames:
            if (altResName, at) in self.patch_charge:
                q.append(self.patch_charge[(altResName, at)])
            elif (resName, at) in self.charge:
                q.append(self.charge[(resName, at)])
            else:
                q.append(0.0)
                warnings.warn(f"Atom type {at} not found for "
                              f"resType {resName} or patch type {altResName}. "
                              f"Set charge to 0.0.")
        return q

    ####################################################################
    #
    #   Simple charges
    #
    ####################################################################

    def evaluate_charges(self, extend_contact_to_residue=False):
        """Evaluate the charges.

        Args:
            extend_contact_to_residue (bool, optional): extend to res
        """
        if self.verbose:
            print('-- Compute list charge for contact atoms only')

        # extract information from the pdb2sq
        xyz = np.array(self.sqldb.get('x,y,z'))
        atinfo = self.sqldb.get(self.atom_key)

        charge = np.array(self.sqldb.get('CHARGE'))

        # define the dictionaries
        charge_data = {}
        charge_data_xyz = {}

        # entire residue or not
        if extend_contact_to_residue:
            indA, indB = self._extend_contact_to_residue()
            index_contact_atoms = indA + indB
        else:
            index_contact_atoms = self.contact_atoms_A + self.contact_atoms_B

        # loop over the chain A
        for i in index_contact_atoms:

            # atinfo
            key = tuple(atinfo[i])

            # store in the dicts
            charge_data[key] = [charge[i]]

            # xyz format
            chain_dict = [{self.chain1: 0, self.chain2: 1}[key[0]]]
            key = tuple(chain_dict + xyz[i, :].tolist())
            charge_data_xyz[key] = [charge[i]]

        # add the electrosatic feature
        self.feature_data['charge'] = charge_data
        self.feature_data_xyz['charge'] = charge_data_xyz

    ####################################################################
    #
    #   PAIR INTERACTIONS
    #
    ####################################################################

    def evaluate_pair_interaction(self, print_interactions=False,
                                  save_interactions=False):
        """Evalaute the pair interactions (coulomb and vdw).

        Args:
            print_interactions (bool, optional): print data to screen
            save_interactions (bool, optional): save the itneractions to file.
        """

        if self.verbose:
            print('-- Compute interaction energy for contact pairs only')

        # extract information from the pdb2sq
        xyz = np.array(self.sqldb.get('x,y,z'))
        atinfo = self.sqldb.get(self.atom_key)

        charge = np.array(self.sqldb.get('CHARGE'))
        vdw = np.array(self.sqldb.get('eps,sig'))
        eps, sig = vdw[:, 0], vdw[:, 1]

        # define the dictionaries
        # these holds data like
        # chainID resname resSeq,name values
        electro_data = {}
        vdw_data = {}

        # define the dict that hold
        #  x y z values
        electro_data_xyz = {}
        vdw_data_xyz = {}

        # define the matrices
        natA, natB = len(self.sqldb.get('x', chainID=self.chain1)), len(
            self.sqldb.get('x', chainID=self.chain2))
        matrix_elec = np.zeros((natA, natB))
        matrix_vdw = np.zeros((natA, natB))

        # handle the export of the interaction breakdown
        _save_ = False
        if save_interactions:
            if save_interactions:
                save_interactions = './'
            if os.path.isdir(save_interactions):
                fname = os.path.join(save_interactions,
                                     'atomic_pair_interaction.dat')
            else:
                fname = save_interactions
            f = open(fname, 'w')
            _save_ = True

        # total energy terms
        ec_tot, evdw_tot = 0, 0

        # loop over the chain A
        for iA, indsB in self.contact_pairs.items():

            # coulomb terms
            r = np.sqrt(np.sum((xyz[indsB, :] - xyz[iA, :])**2, 1))
            r[r == 0] = 3.0
            q1q2 = charge[iA] * charge[indsB]
            ec = q1q2 * self.c / (self.eps0 * r) * \
                (1 - (r / self.contact_cutoff)**2) ** 2

            # coulomb terms
            sigma_avg = 0.5 * (sig[iA] + sig[indsB])
            eps_avg = np.sqrt(eps[iA] * eps[indsB])

            # normal LJ potential
            evdw = 4.0 * eps_avg * \
                ((sigma_avg / r)**12 - (sigma_avg / r)**6) * self._prefactor_vdw(r)

            # total energy terms
            ec_tot += np.sum(ec)
            evdw_tot += np.sum(evdw)

            # atinfo
            keyA = tuple(atinfo[iA])

            # store in matrix form so that
            # we don't have to recalculate for B
            # here assumes that the chainID order is A,B...
            # otherwise rowID will be different with the matrix index
            indb_matrix = [i - natA for i in indsB]
            matrix_elec[iA, indb_matrix] = ec
            matrix_vdw[iA, indb_matrix] = evdw

            # store in the dicts
            electro_data[keyA] = [np.sum(ec)]
            vdw_data[keyA] = [np.sum(evdw)]

            # store in the xyz dict
            key = tuple([0] + xyz[iA, :].tolist())
            electro_data_xyz[key] = [np.sum(ec)]
            vdw_data_xyz[key] = [np.sum(evdw)]

            # print the result
            if _save_ or print_interactions:

                for iB, indexB in enumerate(indsB):

                    line = ''
                    keyB = tuple(atinfo[indexB])

                    line += '{:<3s}'.format(keyA[0])
                    line += '\t{:>1d}'.format(keyA[1])
                    line += '\t{:>4s}'.format(keyA[2])
                    line += '\t{:^4s}'.format(keyA[3])

                    line += '\t{:<3s}'.format(keyB[0])
                    line += '\t{:>1d}'.format(keyB[1])
                    line += '\t{:>4s}'.format(keyB[2])
                    line += '\t{:^4s}'.format(keyB[3])

                    line += '\t{: 6.3f}'.format(r[iB])
                    line += '\t{: f}'.format(ec[iB])
                    line += '\t{: e}'.format(evdw[iB])

                    # print and/or save the interactions
                    if print_interactions:
                        print(line)

                    if _save_:
                        line += '\n'
                        f.write(line)

        # print the total interactions
        if print_interactions or _save_:
            line = '\n\n'
            line += 'Total Evdw  = {:> 12.8f}\n'.format(evdw_tot)
            line += 'Total Eelec = {:> 12.8f}\n'.format(ec_tot)
            if print_interactions:
                print(line)
            if _save_:
                f.write(line)

        # close export file
        if _save_:
            f.close()
            print(f'AtomicFeature coulomb and vdw exported to file {fname}')

        # loop over the B atoms
        for indexB in self.contact_atoms_B:

            # atinfo
            keyB = tuple(atinfo[indexB])

            # extract the values from the matrix
            ec = matrix_elec[:, indexB - natA]
            evdw = matrix_vdw[:, indexB - natA]

            # store in the dict
            electro_data[keyB] = [np.sum(ec)]
            vdw_data[keyB] = [np.sum(evdw)]

            # store the xyz dict
            key = tuple([1] + xyz[indexB, :].tolist())
            electro_data_xyz[key] = [np.sum(ec)]
            vdw_data_xyz[key] = [np.sum(evdw)]

        # add the electrosatic feature
        self.feature_data['coulomb'] = electro_data
        self.feature_data_xyz['coulomb'] = electro_data_xyz

        # add the vdw feature
        self.feature_data['vdwaals'] = vdw_data
        self.feature_data_xyz['vdwaals'] = vdw_data_xyz

    ####################################################################
    #
    #   ELECTROSTATIC
    #
    ####################################################################

    def compute_coulomb_interchain_only(self, dosum=True, contact_only=False):
        """Compute the coulomb interactions between the chains only.

        Args:
            dosum (bool, optional): sum the interaction terms for each atoms
            contact_only (bool, optional): consider only contact atoms
        """

        if self.verbose:
            print('-- Compute coulomb energy interchain only')

        if contact_only:

            xyzA = np.array(self.sqldb.get(
                'x,y,z', rowID=self.contact_atoms_A))
            xyzB = np.array(self.sqldb.get(
                'x,y,z', rowID=self.contact_atoms_B))

            chargeA = np.array(self.sqldb.get(
                'CHARGE', rowID=self.contact_atoms_A))
            chargeB = np.array(self.sqldb.get(
                'CHARGE', rowID=self.contact_atoms_B))

            atinfoA = self.sqldb.get(
                self.atom_key, rowID=self.contact_atoms_A)
            atinfoB = self.sqldb.get(
                self.atom_key, rowID=self.contact_atoms_B)

        else:

            xyzA = np.array(self.sqldb.get('x,y,z', chainID=self.chain1))
            xyzB = np.array(self.sqldb.get('x,y,z', chainID=self.chain2))

            chargeA = np.array(self.sqldb.get('CHARGE', chainID=self.chain1))
            chargeB = np.array(self.sqldb.get('CHARGE', chainID=self.chain2))

            atinfoA = self.sqldb.get(
                self.atom_key, chainID=self.chain1)
            atinfoB = self.sqldb.get(
                self.atom_key, chainID=self.chain2)

        natA, natB = len(xyzA), len(xyzB)
        matrix = np.zeros((natA, natB))

        electro_data = {}

        for iat in range(natA):

            # coulomb terms
            r = np.sqrt(np.sum((xyzB - xyzA[iat, :])**2, 1))
            r[r == 0] = 3.0
            q1q2 = chargeA[iat] * chargeB
            value = q1q2 / r

            # store amd symmtrized these values
            matrix[iat, :] = value

            # atinfo
            key = tuple(atinfoA[iat])

            # store
            if dosum:
                value = [np.sum(value)]
            electro_data[key] = value

        for iat in range(natB):

            # atinfo
            key = tuple(atinfoB[iat])

            # store
            value = matrix[:, iat]
            if dosum:
                value = [np.sum(value)]
            electro_data[key] = value

    ####################################################################
    #
    #   VAN DER WAALS
    #
    ####################################################################

    def compute_vdw_interchain_only(self, dosum=True, contact_only=False):
        """Compute the vdw interactions between the chains only.

        Args:
            dosum (bool, optional): sum the interaction terms for each atoms
            contact_only (bool, optional): consider only contact atoms
        """
        if self.verbose:
            print('-- Compute vdw energy interchain only')

        if contact_only:

            xyzA = np.array(self.sqldb.get(
                'x,y,z', rowID=self.contact_atoms_A))
            xyzB = np.array(self.sqldb.get(
                'x,y,z', rowID=self.contact_atoms_B))

            vdwA = np.array(self.sqldb.get(
                'eps,sig', rowID=self.contact_atoms_A))
            vdwB = np.array(self.sqldb.get(
                'eps,sig', rowID=self.contact_atoms_B))

            epsA, sigA = vdwA[:, 0], vdwA[:, 1]
            epsB, sigB = vdwB[:, 0], vdwB[:, 1]

            atinfoA = self.sqldb.get(
                self.atom_key, rowID=self.contact_atoms_A)
            atinfoB = self.sqldb.get(
                self.atom_key, rowID=self.contact_atoms_B)

        else:

            xyzA = np.array(self.sqldb.get('x,y,z', chainID=self.chain1))
            xyzB = np.array(self.sqldb.get('x,y,z', chainID=self.chain2))

            vdwA = np.array(self.sqldb.get('eps,sig', chainID=self.chain1))
            vdwB = np.array(self.sqldb.get('eps,sig', chainID=self.chain2))

            epsA, sigA = vdwA[:, 0], vdwA[:, 1]
            epsB, sigB = vdwB[:, 0], vdwB[:, 1]

            atinfoA = self.sqldb.get(
                self.atom_key, chainID=self.chain1)
            atinfoB = self.sqldb.get(
                self.atom_key, chainID=self.chain2)

        natA, natB = len(xyzA), len(xyzB)
        matrix = np.zeros((natA, natB))

        vdw_data = {}

        for iat in range(natA):

            # vdW terms
            r = np.sqrt(np.sum((xyzB - xyzA[iat, :])**2, 1))
            r[r == 0] = 3.0
            sigma = 0.5 * (sigA[iat] + sigB)
            eps = np.sqrt(epsA[iat] * epsB)

            # normal LJ potential
            value = 4 * eps * ((sigma / r)**12 - (sigma / r)**6)

            # store these values
            matrix[iat, :] = value

            # atinfo
            key = tuple(atinfoA[iat])

            # store
            if dosum:
                value = [np.sum(value)]
            vdw_data[key] = value

        for iat in range(natB):

            # atinfo
            key = tuple(atinfoB[iat])

            # store
            value = matrix[:, iat]
            if dosum:
                value = [np.sum(value)]
            vdw_data[key] = value

    @staticmethod
    def _prefactor_vdw(r):
        """prefactor for vdw interactions."""

        r_off, r_on = 8.5, 6.5
        r2 = r**2
        pref = (r_off**2 - r2)**2 * (r_off**2 - r2 - 3 *
                                     (r_on**2 - r2)) / (r_off**2 - r_on**2)**3
        pref[r > r_off] = 0.
        pref[r < r_on] = 1.0
        return pref


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
    path = os.path.dirname(os.path.realpath(__file__))
    FF = path + '/forcefield/'

    atfeat = AtomicFeature(pdb_data,
                           chain1=chain1,
                           chain2=chain2,
                           param_charge=FF + 'protein-allhdg5-4_new.top',
                           param_vdw=FF + 'protein-allhdg5-4_new.param',
                           patch_file=FF + 'patch.top')

    atfeat.assign_parameters()

    # only compute the pair interactions here
    atfeat.evaluate_pair_interaction(print_interactions=False)

    # compute the charges
    # here we extand the contact atoms to
    # entire residue containing at least 1 contact atom
    atfeat.evaluate_charges(extend_contact_to_residue=True)

    # export in the hdf5 file
    atfeat.export_dataxyz_hdf5(featgrp)
    atfeat.export_data_hdf5(featgrp_raw)

    # close
    atfeat.sqldb._close()


########################################################################
#
#  TEST THE CLASS
#
########################################################################

if __name__ == '__main__':

    from pprint import pprint
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.realpath(__file__))))
    pdb_file = os.path.join(base_path, "test/1AK4/native/1AK4.pdb")
    FF = os.path.join(base_path, 'deeprank/features/forcefield/')

    atfeat = AtomicFeature(pdb_file,
                           param_charge=FF + 'protein-allhdg5-4_new.top',
                           param_vdw=FF + 'protein-allhdg5-4_new.param',
                           patch_file=FF + 'patch.top',
                           verbose=True)

    atfeat.assign_parameters()
    atfeat.evaluate_pair_interaction()
    atfeat.evaluate_charges(extend_contact_to_residue=True)
    atfeat.sqldb._close()

    # export in the hdf5 file
    pprint(atfeat.feature_data)
    print()
    pprint(atfeat.feature_data_xyz)
