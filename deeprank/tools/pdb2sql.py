import os
import sqlite3
import subprocess as sp
import sys
import warnings

import numpy as np

from deeprank.config import logger


class pdb2sql(object):

    def __init__(self,
                 pdbfile,
                 sqlfile=None,
                 fix_chainID=True,
                 verbose=False,
                 no_extra=True):
        """Create a SQL data base for a PDB file.

        This allows to easily parse and extract information of the PDB
        using SQL queries. This is a local version of the pdb2sql tool
        (https://github.com/DeepRank/pdb2sql). pdb2sql is further
        developped as a standalone we should use the library directly.

        Args:
            pdbfile (str or list(bytes)) : name of pdbfile or
                list of bytes containing the pdb data
            sqlfile (str, optional): name of the sqlfile.
                By default it is created in memory only.
            fix_chainID (bool, optinal): check if the name of the chains
                are A,B,C, .... and fix it if not.
            verbose (bool): probably print stuff.
            no_extra (bool): remove occupancy and tempFactor clumns or not

        Examples:
            >>> # create the sql
            >>> db = pdb2sql('1AK4_100w.pdb')
            >>>
            >>> # print the database
            >>> db.prettyprint()
            >>>
            >>> # get the names of the columns
            >>> db.get_colnames()
            >>>
            >>> # extract the xyz position of the atoms with name CB
            >>> xyz = db.get('*',index=[0,1,2,3])
            >>> print(xyz)
            >>>
            >>> xyz = db.get('rowID',where="resName='VAL'")
            >>> print(xyz)
            >>>
            >>> db.add_column('CHARGE','FLOAT')
            >>> db.put('CHARGE',0.1)
            >>> db.prettyprint()
            >>>
            >>> db.exportpdb('chainA.pdb',where="chainID='A'")
            >>>
            >>> # close the database
            >>> db.close()
        """
        self.pdbfile = pdbfile
        self.sqlfile = sqlfile
        self.is_valid = True
        self.verbose = verbose
        self.no_extra = no_extra

        # create the database
        self._create_sql()

        # backbone type
        self.backbone_type = ['C', 'CA', 'N', 'O']

        # hard limit for the number of SQL varaibles
        self.SQLITE_LIMIT_VARIABLE_NUMBER = 999
        self.max_sql_values = 950

        # fix the chain ID
        if fix_chainID:
            self._fix_chainID()

        # a few constant
        self.residue_key = 'chainID,resSeq,resName'
        # self.atom_key = 'chainID,resSeq,resName,name'
    ####################################################################
    #
    #   CREATION AND PRINTING
    #
    ####################################################################

    def _create_sql(self):
        """Create the sql database."""

        pdbfile = self.pdbfile
        sqlfile = self.sqlfile

        if self.verbose:
            logger.info('-- Create SQLite3 database')

        # name of the table
        # table = 'ATOM'

        # column names and types
        self.col = {'serial': 'INT',
                    'name': 'TEXT',
                    'altLoc': 'TEXT',
                    'resName': 'TEXT',
                    'chainID': 'TEXT',
                    'resSeq': 'INT',
                    'iCode': 'TEXT',
                    'x': 'REAL',
                    'y': 'REAL',
                    'z': 'REAL',
                    'occ': 'REAL',
                    'temp': 'REAL'
                    }

        # delimtier of the column format
        # taken from
        # http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
        self.delimiter = {
            'serial': [6, 11],
            'name': [12, 16],
            'altLoc': [16, 17],
            'resName': [17, 20],
            'chainID': [21, 22],
            'resSeq': [22, 26],
            'iCode': [26, 27],
            'x': [30, 38],
            'y': [38, 46],
            'z': [46, 54],
            'occ': [54, 60],
            'temp': [60, 66]}

        if self.no_extra:
            del self.col['occ']
            del self.col['temp']

        # size of the things
        ncol = len(self.col)

        # open the data base
        # if we do not specify a db name
        # the db is only in RAM
        # there might be little advantage to use memory
        # https://stackoverflow.com/questions/764710/
        if self.sqlfile is None:
            self.conn = sqlite3.connect(':memory:')

        # or we create a new db file
        else:
            if os.path.isfile(sqlfile):
                sp.call('rm %s' % sqlfile, shell=True)
            self.conn = sqlite3.connect(sqlfile)
        self.c = self.conn.cursor()

        # intialize the header/placeholder
        header, qm = '', ''
        for ic, (colname, coltype) in enumerate(self.col.items()):
            header += f'{colname} {coltype}'
            qm += '?'
            if ic < ncol - 1:
                header += ', '
                qm += ','

        # create the table
        query = f'CREATE TABLE ATOM ({header})'
        self.c.execute(query)

        # read the pdb file
        # this is dangerous if there are ATOM written in the comment part
        # which happends often
        # data = sp.check_output("awk '/ATOM/' %s" %pdbfile,shell=True).decode('utf8').split('\n')

        # a safer version consist at matching against the first field
        # won't work on windows
        # data = sp.check_output("awk '$1 ~ /^ATOM/' %s" %pdbfile,shell=True).decode('utf8').split('\n')

        # a pure python way
        # RMK we go through the data twice here. Once to read the ATOM line and once to parse the data ...
        # we could do better than that. But the most time consuming step seems to be the CREATE TABLE query
        # if we path a file we read it
        if isinstance(pdbfile, str):
            if os.path.isfile(pdbfile):
                with open(pdbfile, 'r') as fi:
                    data = [line.split('\n')[0]
                            for line in fi if line.startswith('ATOM')]
            else:
                raise FileNotFoundError(f'PDB file {pdbfile} not found')

        # if we pass a list as for h5py read/write
        # we directly use that
        elif isinstance(pdbfile, np.ndarray):
            data = [l.decode('utf-8') for l in pdbfile.tolist()]

        # if we cant read it
        else:
            raise ValueError(f'PDB data not recognized: {pdbfile}')

        # if there is no ATOM in the file
        if len(data) == 1 and data[0] == '':
            self.is_valid = False
            raise ValueError(f'No ATOM found in the pdb data {pdbfile}')

        # haddock chain ID fix
        del_copy = self.delimiter.copy()
        if data[0][del_copy['chainID'][0]] == ' ':
            del_copy['chainID'] = [72, 73]

        # get all the data
        data_atom = []
        for line in data:

            # sometimes we still have an empty line somewhere
            if len(line) == 0:
                continue

            # browse all attribute of each atom
            at = ()
            for (colname, coltype) in self.col.items():

                # get the piece of data
                data_col = line[del_copy[colname][0]:
                                del_copy[colname][1]].strip()

                # convert it if necessary
                if coltype == 'INT':
                    data_col = int(data_col)
                elif coltype == 'REAL':
                    data_col = float(data_col)

                # append keep the comma !!
                # we need proper tuple
                at += (data_col,)

            # append
            data_atom.append(at)

        # push in the database
        self.c.executemany(f'INSERT INTO ATOM VALUES ({qm})', data_atom)

    def _fix_chainID(self):
        """Fix the chain ID if necessary.

        Replace the chain ID by A,B,C,D, ..... in that order
        """

        from string import ascii_uppercase

        # get the current names
        data = self.get('chainID')
        natom = len(data)

        # get uniques
        chainID = []
        for c in data:
            if c not in chainID:
                chainID.append(c)

        if chainID == ['A', 'B']:
            return

        if len(chainID) > 26:
            warnings.warn(
                f"More than 26 chains have been detected. "
                f"This is so far not supported")
            sys.exit()

        # declare the new names
        newID = [''] * natom

        # fill in the new names
        for ic, chain in enumerate(chainID):
            index = self.get('rowID', chainID=chain)
            for ind in index:
                newID[ind] = ascii_uppercase[ic]

        # update the new name
        self.update_column('chainID', newID)

    # get the names of the columns
    def get_colnames(self):
        """Print the colom names of the database."""

        cd = self.conn.execute('select * from atom')
        print('Possible column names are:')
        names = list(map(lambda x: x[0], cd.description))
        print('\trowID')
        for n in names:
            print('\t' + n)

    # print the database
    def prettyprint(self):
        """Print the database with pandas."""

        import pandas.io.sql as psql
        df = psql.read_sql("SELECT * FROM ATOM", self.conn)
        print(df)

    def uglyprint(self):
        """Raw print of the database."""

        ctmp = self.conn.cursor()
        ctmp.execute("SELECT * FROM ATOM")
        print(ctmp.fetchall())

    ####################################################################
    #
    # GET FUNCTIONS
    #
    # get(attribute,selection) -> return the atribute(s) value(s)
    #                               for the given selection
    # get_contact_atoms()      -> return a list of rowID
    #                               for the contact atoms
    # get_contact_residue()    -> return a list of resSeq
    #                               for the contact residue
    #
    ####################################################################

    def get(self, atnames, **kwargs):
        """Get data from the sql database.

        Get the values of specified attributes for a specific selection.

        Args:

            atnames (str): attribute name. They can be printed via
                the get_colnames().
                - serial
                - name
                - atLoc
                - resName
                - chainID
                - resSeq,
                - iCode,
                - x/y/z
                Several attributes can be specified at once e.g 'x,y,z'

            kwargs : Several options are possible to select atoms.
                Each column can be used as a keyword argument.
                Several keywords can be combined assuming a AND
                logical combination.
                    None : return the entire table
                    chainID = 'A' select chain from name
                    resIndex = [1,2,3] select residue from index
                    resName = ['VAL', 'LEU'] select residue from name
                    name  = ['CA', 'N'] select atoms from names
                    rowID = [1,2,3] select atoms from index

        Returns:
           np.array:  Numpy array containing the requested data.

        Examples:
            >>> db = pdb2sql(filename)
            >>> xyz = db.get('x,y,z',name = ['CA', 'CB'])
            >>> xyz = db.get('x,y,z',chainID='A',resName=['VAL', 'LEU'])
        """

        # the asked keys
        keys = kwargs.keys()

        # check if the column exists
        try:
            self.c.execute(f"SELECT EXISTS(SELECT {atnames} FROM ATOM)")
        except BaseException:
            logger.error(
                f"Column {atnames} not found in the database")
            self.get_colnames()
            return

        # if we have 0 key we take the entire db
        if len(kwargs) == 0:
            query = 'SELECT {an} FROM ATOM'.format(an=atnames)
            data = [list(row) for row in self.c.execute(query)]

        ################################################################
        # GENERIC QUERY
        #
        # the only one we need
        # each keys must be a valid columns
        # each value may be a single value or an array
        # AND is assumed between different keys
        # OR is assumed for the different values of a given key
        #
        ################################################################
        else:

            # check that all the keys exists
            for k in keys:

                if k.startswith('no_'):
                    k = k[3:]

                try:
                    self.c.execute(f"SELECT EXISTS(SELECT {k} FROM ATOM)")
                except BaseException:
                    logger.error(f'Column {k} not found in the database')
                    self.get_colnames()
                    return

            # form the query and the tuple value
            query = 'SELECT {an} FROM ATOM WHERE '.format(an=atnames)
            conditions = []
            vals = ()

            # iterate through the kwargs
            for (k, v) in kwargs.items():

                # deals with negative conditions
                if k.startswith('no_'):
                    k = k[3:]
                    neg = ' NOT'
                else:
                    neg = ''

                # get if we have an array or a scalar
                # and build the value tuple for the sql query
                # deal with the indexing issue if rowID is required
                if isinstance(v, list):

                    nv = len(v)

                    # if we have a large number of values
                    # we must cut that in pieces because SQL has a hard limit
                    # that is 999. The limit is here set to 950
                    # so that we can have multiple conditions with a total number
                    # of values inferior to 999
                    if nv > self.max_sql_values:

                        # cut in chunck
                        chunck_size = self.max_sql_values
                        vchunck = [v[i:i + chunck_size]
                                   for i in range(0, nv, chunck_size)]

                        data = []
                        for v in vchunck:
                            new_kwargs = kwargs.copy()
                            new_kwargs[k] = v
                            data += self.get(atnames, **new_kwargs)
                        return data

                    # otherwithe we just go on
                    else:

                        if k == 'rowID':
                            vals = vals + tuple([iv + 1 for iv in v])
                        else:
                            vals = vals + tuple(v)

                else:

                    nv = 1
                    if k == 'rowID':
                        vals = vals + (v + 1,)
                    else:
                        vals = vals + (v,)

                # create the condition for that key
                conditions.append(k + neg + ' in (' + ','.join('?' * nv) + ')')

            # stitch the conditions and append to the query
            query += ' AND '.join(conditions)

            # error if vals is too long
            if len(vals) > self.SQLITE_LIMIT_VARIABLE_NUMBER:
                logger.error(
                    f'\n  SQL Queries can only handle a total of 999 values.'
                    f'\n  The current query has {len(vals)} values'
                    f'\n  Hence it fails.'
                    f'\n  You are in a rare situation where MULTIPLE '
                    f'conditions have a combined number of values that is '
                    f'too large'
                    f'\n These conditions are:')
                ntot = 0
                for k, v in kwargs.items():
                    logger.error(f'\n    : --> {k:10s} : {len(v)} values.')
                    ntot += len(v)
                logger.error(f'\n    : --> %10s : %d values' % ('Total', ntot))
                logger.error(
                    f'\n    : Try to decrease self.max_sql_values '
                    f'in pdb2sql.py\n')
                raise ValueError('Too many SQL variables')

            # query the sql database and return the answer in a list
            data = [list(row) for row in self.c.execute(query, vals)]

        # empty data
        if len(data) == 0:
            warnings.warn('sqldb.get returned an empty')
            return data

        # fix the python <--> sql indexes
        # if atnames == 'rowID':
        if 'rowID' in atnames:
            index = atnames.split(',').index('rowID')
            for i, _ in enumerate(data):
                data[i][index] -= 1

        # postporcess the output of the SQl query
        # flatten it if each els is of size 1
        if len(data[0]) == 1:
            data = [d[0] for d in data]

        return data

    ####################################################################
    #
    # get the contact atoms
    #
    # we should have a entire module called pdb2sql
    # with a submodule pdb2sql.interface that finds
    # contact atoms/residues,
    # and possbily other submodules to do other things
    # that will leave only the get / put methods in the main class
    #
    ####################################################################
    def get_contact_atoms(self,
                          cutoff=8.5,
                          chain1='A',
                          chain2='B',
                          extend_to_residue=False,
                          only_backbone_atoms=False,
                          excludeH=False,
                          return_only_backbone_atoms=False,
                          return_contact_pairs=False):
        """Get contact atoms of the interface.

        The cutoff distance is by default 8.5 Angs but can be changed
        at will. A few more options allows to precisely define
        how the contact atoms are identified and returned.

        Args:
            cutoff (float): cutoff for contact atoms (default 8.5)
            chain1 (str): name of the first chain
            chain2 (str): name of the first chain
            extend_to_residue (bool): extend the contact atoms to
                entire residues.
            only_bacbone_atoms (bool): consider only backbone atoms
            excludeH (bool): exclude hydrogen atoms
            return_only_backbone_atoms (bool): only returns backbone atoms
            return_contact_pairs (bool): return the contact pairs
                instead of contact atoms.

        Raises:
            ValueError: contact atoms not found.

        Returns:
            np.array: index of the contact atoms

        Examples:

            >>> db = pdb2sql(filename)
            >>> db.get_contact_atoms(cutoff=5.0,return_contact_pairs=True)
        """
        # xyz of the chains
        xyz1 = np.array(self.get('x,y,z', chainID=chain1))
        xyz2 = np.array(self.get('x,y,z', chainID=chain2))

        # index of b
        index2 = self.get('rowID', chainID=chain2)

        # resName of the chains
        # resName1 = np.array(self.get('resName', chainID=chain1))
        # resName2 = np.array(self.get('resName',chainID=chain2))

        # atomnames of the chains
        atName1 = np.array(self.get('name', chainID=chain1))
        atName2 = np.array(self.get('name', chainID=chain2))

        # loop through the first chain
        # TO DO : loop through the smallest chain instead ...
        index_contact_1, index_contact_2 = [], []
        index_contact_pairs = {}

        for i, x0 in enumerate(xyz1):

            # compute the contact atoms
            contacts = np.where(
                np.sqrt(
                    np.sum(
                        (xyz2 - x0)**2,
                        1)) <= cutoff)[0]

            # exclude the H if required
            if excludeH and atName1[i][0] == 'H':
                continue

            if len(contacts) > 0 and any([not only_backbone_atoms,
                                          atName1[i] in self.backbone_type]):

                # the contact atoms
                index_contact_1 += [i]
                index_contact_2 += [
                    index2[k] for k in contacts if
                    (any([atName2[k] in self.backbone_type,
                          not only_backbone_atoms])
                     and not (excludeH and atName2[k][0] == 'H'))]

                # the pairs
                pairs = [
                    index2[k] for k in contacts if
                    any([atName2[k] in self.backbone_type,
                         not only_backbone_atoms])
                    and not (excludeH and atName2[k][0] == 'H')]
                if len(pairs) > 0:
                    index_contact_pairs[i] = pairs

        # get uniques
        index_contact_1 = sorted(set(index_contact_1))
        index_contact_2 = sorted(set(index_contact_2))

        # if no atoms were found
        if len(index_contact_1) == 0:
            raise ValueError(f"No contact atoms found with cutoff {cutoff}Å")

        # extend the list to entire residue
        if extend_to_residue:
            index_contact_1, index_contact_2 = self._extend_contact_to_residue(
                index_contact_1, index_contact_2, only_backbone_atoms)

        # filter only the backbone atoms
        if return_only_backbone_atoms and not only_backbone_atoms:

            # get all the names
            # there are better ways to do that !
            atNames = np.array(self.get('name'))

            # change the index_contacts
            index_contact_1 = [ind for ind in index_contact_1
                               if atNames[ind] in self.backbone_type]
            index_contact_2 = [ind for ind in index_contact_2
                               if atNames[ind] in self.backbone_type]

            # change the contact pairs
            tmp_dict = {}
            for ind1, ind2_list in index_contact_pairs.items():

                if atNames[ind1] in self.backbone_type:
                    tmp_dict[ind1] = [ind2 for ind2 in ind2_list
                                      if atNames[ind2] in self.backbone_type]

            index_contact_pairs = tmp_dict

        # not sure that's the best way of dealing with that
        if return_contact_pairs:
            return index_contact_pairs
        else:
            return index_contact_1, index_contact_2

    # extend the contact atoms to the residue
    def _extend_contact_to_residue(self, index1, index2, only_backbone_atoms):

        # extract the data
        dataA = self.get(self.residue_key, rowID=index1)
        dataB = self.get(self.residue_key, rowID=index2)

        # create tuple cause we want to hash through it
        # dataA = list(map(lambda x: tuple(x),dataA))
        # dataB = list(map(lambda x: tuple(x),dataB))
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

            if only_backbone_atoms:
                index_contact_A += self.get('rowID',
                                            chainID=chainID,
                                            resName=resName,
                                            resSeq=resSeq,
                                            name=self.backbone_type)
            else:
                index_contact_A += self.get('rowID',
                                            chainID=chainID,
                                            resName=resName,
                                            resSeq=resSeq)

        # contact of chain B
        for resdata in resB:
            chainID, resSeq, resName = resdata

            if only_backbone_atoms:
                index_contact_B += self.get('rowID',
                                            chainID=chainID,
                                            resName=resName,
                                            resSeq=resSeq,
                                            name=self.backbone_type)
            else:
                index_contact_B += self.get('rowID',
                                            chainID=chainID,
                                            resName=resName,
                                            resSeq=resSeq)

        # make sure that we don't have double (maybe optional)
        index_contact_A = sorted(set(index_contact_A))
        index_contact_B = sorted(set(index_contact_B))

        return index_contact_A, index_contact_B

    # get the contact residue
    def get_contact_residue(self,
                            cutoff=8.5,
                            chain1='A',
                            chain2='B',
                            excludeH=False,
                            only_backbone_atoms=False,
                            return_contact_pairs=False):
        """Get contact residues of the interface.

        The cutoff distance is by default 8.5 Angs but can be changed
        at will. A few more options allows to precisely define how
        the contact residues are identified and returned.

        Args:
            cutoff (float): cutoff for contact atoms (default 8.5)
            chain1 (str): name of the first chain
            chain2 (str): name of the first chain
            only_bacbone_atoms (bool): consider only backbone atoms
            excludeH (bool): exclude hydrogen atoms
            return_contact_pairs (bool): return the contact pairs
                instead of contact atoms

        Returns:
            np.array: index of the contact atoms

        Examples:
            >>> db = pdb2sql(filename)
            >>> db.get_contact_residue(cutoff=5.0,
            ... return_contact_pairs=True)
        """
        # get the contact atoms
        if return_contact_pairs:

            # declare the dict
            residue_contact_pairs = {}

            # get the contact atom pairs
            atom_pairs = self.get_contact_atoms(
                cutoff=cutoff, chain1=chain1, chain2=chain2,
                only_backbone_atoms=only_backbone_atoms,
                excludeH=excludeH,
                return_contact_pairs=True)

            # loop over the atom pair dict
            for iat1, atoms2 in atom_pairs.items():

                # get the res info of the current atom
                data1 = tuple(self.get(self.residue_key, rowID=[iat1])[0])

                # create a new entry in the dict if necessary
                if data1 not in residue_contact_pairs:
                    residue_contact_pairs[data1] = set()

                # get the res info of the atom in the other chain
                data2 = self.get(self.residue_key, rowID=atoms2)

                # store that in the dict without double
                for resData in data2:
                    residue_contact_pairs[data1].add(tuple(resData))

            for resData in residue_contact_pairs.keys():
                residue_contact_pairs[resData] = sorted(
                    residue_contact_pairs[resData])

            return residue_contact_pairs

        else:

            # get the contact atoms
            contact_atoms = self.get_contact_atoms(
                cutoff=cutoff, chain1=chain1, chain2=chain2,
                return_contact_pairs=False)

            # get the residue info
            data1 = self.get(self.residue_key, rowID=contact_atoms[0])
            data2 = self.get(self.residue_key, rowID=contact_atoms[1])

            # take only unique
            residue_contact_A = sorted(
                set([tuple(resData) for resData in data1]))
            residue_contact_B = sorted(
                set([tuple(resData) for resData in data2]))

            return residue_contact_A, residue_contact_B

    ####################################################################
    #
    #       PUT FUNCTONS AND ASSOCIATED
    #
    #           add_column()    -> add a column
    #           update_column() -> update the values of one column
    #           update_xyz()    -> update_xyz of the pdb
    #           put()           -> put a value in a column
    #
    ####################################################################

    def add_column(self, colname, coltype='FLOAT', default=0):
        """Add an extra column to the ATOM table.

        Args:
            colname (str): name of the column
            coltype (str, optional): type of the column data
                (default FLOAT)
            default (int, optional): default value to fill in the column
                (default 0.0)
        """

        query = "ALTER TABLE ATOM ADD COLUMN '%s' %s DEFAULT %s" % (
            colname, coltype, str(default))
        self.c.execute(query)

    def update(self, attribute, values, **kwargs):
        """Update multiple columns in the data.

        Args:
            attribute (str): comma separated attribute names: 'x,y,z'
            values (np.array): new values for the attributes
            **kwargs: selection of the rows to update.

        Raises:
            ValueError: if size mismatch between values, conditions
                and attribute names

        Examples:
            >>> n = 200
            >>> index = list(range(n))
            >>> vals = np.random.rand(n,3)
            >>> db.update('x,y,z',vals,rowID=index)
        """

        # the asked keys
        # keys = kwargs.keys()

        # check if the column exists
        try:
            self.c.execute(f"SELECT EXISTS(SELECT {attribute} FROM ATOM)")
        except BaseException:
            logger.error(f'Column {attribute} not found in the database')
            self.get_colnames()
            raise ValueError(f'Attribute name {attribute} not recognized')

        # if len(kwargs) == 0:
        #    raise ValueError(f'Update without kwargs seem to be buggy.'
        #                     f' Use rowID=list(range(natom)) instead')

        # handle the multi model cases
        # this is still in devs and not necessary
        # for deeprank.
        # We will have to deal with that if we
        # release pdb2sql as a standalone
        # if 'model' not in keys and self.nModel > 0:
        #     for iModel in range(self.nModel):
        #         kwargs['model'] = iModel
        #         self.update(attribute,values,**kwargs)
        #     return

        # parse the attribute
        if ',' in attribute:
            attribute = attribute.split(',')

        if not isinstance(attribute, list):
            attribute = [attribute]

        # check the size
        natt = len(attribute)
        nrow = len(values)
        ncol = len(values[0])

        if natt != ncol:
            raise ValueError(
                f'Number of attribute incompatible with '
                f' the number of columns in the data')

        # get the row ID of the selection
        rowID = self.get('rowID', **kwargs)
        nselect = len(rowID)

        if nselect != nrow:
            raise ValueError(
                'Number of data values incompatible with the given conditions')

        # prepare the query
        query = 'UPDATE ATOM SET '
        query = query + ', '.join(map(lambda x: x + '=?', attribute))
        # if len(kwargs)>0: # why did I do that ...
        query = query + ' WHERE rowID=?'

        # prepare the data
        data = []
        for i, val in enumerate(values):

            tmp_data = [v for v in val]

            # if len(kwargs)>0: Same here why did I do that ?
            # here the conversion of the indexes is a bit annoying
            tmp_data += [rowID[i] + 1]

            data.append(tmp_data)

        self.c.executemany(query, data)

    def update_column(self, colname, values, index=None):
        """Update a single column.

        Args:
            colname (str): name of the column to update
            values (list): new values of the column
            index (None, optional): index of the column to update
                (default all)

        Examples:
            >>> db.update_column('x', np.random.rand(10),
            ... index=list(range(10)))
        """

        if index is None:
            data = [[v, i + 1] for i, v in enumerate(values)]
        else:
            # shouldn't that be ind+1 ?
            data = [[v, ind] for v, ind in zip(values, index)]

        query = 'UPDATE ATOM SET {cn}=? WHERE rowID=?'.format(cn=colname)
        self.c.executemany(query, data)
        # self.conn.commit()

    def update_xyz(self, xyz, index=None):
        """Update the xyz information.

        Update the positions of the atoms selected
        if index=None the position of all the atoms are changed

        Args:
            xyz (np.array): new xyz position
            index (None, list(int)): index of the atom to move

        Examples:
            >>> n = 200
            >>> index = list(range(n))
            >>> vals = np.random.rand(n,3)
            >>> db.update_xyz(vals,index=index)
        """

        if index is None:
            data = [[pos[0], pos[1], pos[2], i + 1]
                    for i, pos in enumerate(xyz)]
        else:
            data = [[pos[0], pos[1], pos[2], ind + 1]
                    for pos, ind in zip(xyz, index)]

        query = 'UPDATE ATOM SET x=?, y=?, z=? WHERE rowID=?'
        self.c.executemany(query, data)

    def put(self, colname, value, **kwargs):
        """Update the value of the attribute with value specified with possible
        selection.

        Args:
            colname (str): must be a valid attribute name.
                you can get these names via the get_colnames():
                    serial, name, atLoc,resName, chainID, resSeq,
                    iCode,x,y,z,occ,temp
                you can specify more than one attribute name at once,
                e.g 'x,y,z'

            keyword args: Several options are possible
                None : put the value in the entire column
                index = [0,1,2,3] in only these indexes (not serial)
                where = "chainID='B'" only for this chain
                query = general SQL Query

        Examples:
            >>> db = pdb2sql(filename)
            >>> db.add_column('CHARGE')
            >>> db.put('CHARGE',1.25,index=[1])
            >>> db.close()
        """
        arguments = {
            'where': "String e.g 'chainID = 'A''",
            'index': "Array e.g. [27,28,30]",
            'name': "'CA' atome name",
            'query': "SQL query e.g. 'WHERE chainID='B' AND resName='ASP' "}

        # the asked keys
        keys = kwargs.keys()

        # if we have more than one key we kill it
        if len(keys) > 1:
            logger.error(f'You can only specify 1 conditional statement '
                         f'for the pdb2sql.put function')
            return

        # check if the column exists
        try:
            self.c.execute(f"SELECT EXISTS(SELECT {colname} FROM ATOM)")
        except BaseException:
            logger.error(f'Column {colname} not found in the database')
            self.get_colnames()
            return

        # if we have 0 key we take the entire db
        if len(kwargs) == 0:
            query = f'UPDATE ATOM SET {colname}=?'
            value = tuple([value])
            self.c.execute(query, value)
            return

        # otherwise we have only one key
        key = list(keys)[0]
        cond = kwargs[key]

        # select which key we have
        if key == 'where':
            query = f'UPDATE ATOM SET {colname}=? WHERE {cond}'
            value = tuple([value])
            self.c.execute(query, value)

        elif key == 'name':
            values = tuple([value, cond])
            query = f'UPDATE ATOM SET {colname}=? WHERE name=?'
            self.c.execute(query, values)

        elif key == 'index':
            values = tuple([value] + [v + 1 for v in cond])
            qm = ','.join(['?' for i in range(len(cond))])
            query = f'UPDATE ATOM SET {colname}=? WHERE rowID in ({qm})'
            self.c.execute(query, values)

        elif key == 'query':
            query = f'UPDATE ATOM SET {colname}=? {cond}'
            value = tuple([value])
            self.c.execute(query, value)

        else:
            logger.error(
                f'Error arguments {key} not supported in pdb2sql.get().'
                f'\nOptions are:\n')
            for posskey, possvalue in arguments.items():
                logger.error(f'\t{posskey}\t\t{possvalue}')
            return

    ####################################################################
    #
    #       COMMIT, EXPORT, CLOSE FUNCTIONS
    #
    ####################################################################

    # comit changes

    def commit(self):
        """Commit the database."""
        self.conn.commit()

    # export to pdb file
    def exportpdb(self, fname, **kwargs):
        """Export a PDB file with kwargs selection.

        Args:
            fname (str): Name of the file
            **kwargs: Selection (see pdb2sql.get())

        Examples:
            >>> db = pdb2sql('1AK4.pdb')
            >>> db.exportpdb('CA.pdb',name='CA')
        """
        # get the data
        data = self.get('*', **kwargs)

        # write each line
        # the PDB format is pretty strict
        # http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
        f = open(fname, 'w')
        for d in data:
            line = 'ATOM  '
            line += '{:>5}'.format(d[0])    # serial
            line += ' '
            line += '{:^4}'.format(d[1])    # name
            line += '{:>1}'.format(d[2])    # altLoc
            line += '{:>3}'.format(d[3])  # resname
            line += ' '
            line += '{:>1}'.format(d[4])    # chainID
            line += '{:>4}'.format(d[5])    # resSeq
            line += '{:>1}'.format(d[6])    # iCODE
            line += '   '
            line += '{: 8.3f}'.format(d[7])  # x
            line += '{: 8.3f}'.format(d[8])  # y
            line += '{: 8.3f}'.format(d[9])  # z

            if not self.no_extra:
                line += '{: 6.2f}'.format(d[10])    # occ
                line += '{: 6.2f}'.format(d[11])    # temp

            line += '\n'

            f.write(line)

        # close
        f.close()

    # close the database
    def close(self, rmdb=True):
        """Close the database.

        Args:
            rmdb (bool, optional): Remove the database file
        """

        if self.sqlfile is None:
            self.conn.close()

        else:

            if rmdb:
                self.conn.close()
                os.system('rm %s' % (self.sqlfile))
            else:
                self.commit()
                self.conn.close()

    ####################################################################
    #
    # Transform the position of the molecule
    #
    ####################################################################

    def translation(self, vect, **kwargs):
        """Translate a part or all of the molecule.

        Args:
            vect (np.array): translation vector
            **kwargs: keyword argument to select the atoms.
                See pdb2sql.get()

        Examples:
            >>> vect = np.random.rand(3)
            >>> db.translation(vect, chainID = 'A')
        """
        xyz = self.get('x,y,z', **kwargs)
        xyz += vect
        self.update('x,y,z', xyz, **kwargs)

    def rotation_around_axis(self, axis, angle, **kwargs):
        """Rotate a molecule around a specified axis.

        Args:
            axis (np.array): axis of rotation
            angle (float): angle of rotation in radian
            **kwargs: keyword argument to select the atoms.
                See pdb2sql.get()

        Returns:
            np.array: center of the molecule

        Examples:
            >>> axis = np.random.rand(3)
            >>> angle = np.random.rand()
            >>> db.rotation_around_axis(axis, angle, chainID = 'B')
        """
        xyz = self.get('x,y,z', **kwargs)

        # get the data
        ct, st = np.cos(angle), np.sin(angle)
        ux, uy, uz = axis

        # get the center of the molecule
        xyz0 = np.mean(xyz, 0)

        # definition of the rotation matrix
        # see https://en.wikipedia.org/wiki/Rotation_matrix
        rot_mat = np.array([
            [ct + ux**2 * (1 - ct), ux * uy * (1 - ct) - uz * st, ux * uz * (1 - ct) + uy * st],
            [uy * ux * (1 - ct) + uz * st, ct + uy**2 * (1 - ct), uy * uz * (1 - ct) - ux * st],
            [uz * ux * (1 - ct) - uy * st, uz * uy * (1 - ct) + ux * st, ct + uz**2 * (1 - ct)]])

        # apply the rotation
        xyz = np.dot(rot_mat, (xyz - xyz0).T).T + xyz0
        self.update('x,y,z', xyz, **kwargs)

        return xyz0

    def rotation_euler(self, alpha, beta, gamma, **kwargs):
        """Rotate a part or all of the molecule from Euler rotation axis.

        Args:
            alpha (float): angle of rotation around the x axis
            beta (float): angle of rotation around the y axis
            gamma (float): angle of rotation around the z axis
            **kwargs: keyword argument to select the atoms.
                See pdb2sql.get()

        Examples:
            >>> a,b,c = np.random.rand(3)
            >>> db.rotation_euler(a,b,c,resName='VAL')
        """
        xyz = self.get('x,y,z', **kwargs)

        # precomte the trig
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)

        # get the center of the molecule
        xyz0 = np.mean(xyz, 0)

        # rotation matrices
        rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
        rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
        rot_mat = np.dot(rx, np.dot(ry, rz))

        # apply the rotation
        xyz = np.dot(rot_mat, (xyz - xyz0).T).T + xyz0

        self.update('x,y,z', xyz, **kwargs)

    def rotation_matrix(self, rot_mat, center=True, **kwargs):
        """Rotate a part or all of the molecule from a rotation matrix.

        Args:
            rot_mat (np.array): 3x3 rotation matrix
            center (bool, optional): center the molecule before
                applying the rotation.
            **kwargs: keyword argument to select the atoms.
                See pdb2sql.get()

        Examples:
            >>> mat = np.random.rand(3,3)
            >>> db.rotation_matrix(mat,chainID='A')
        """
        xyz = self.get('x,y,z', **kwargs)

        if center:
            xyz0 = np.mean(xyz)
            xyz = np.dot(rot_mat, (xyz - xyz0).T).T + xyz0
        else:
            xyz = np.dot(rot_mat, (xyz).T).T
        self.update('x,y,z', xyz, **kwargs)
