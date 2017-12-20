import sqlite3
import subprocess as sp 
import os
import numpy as np
from time import time

'''
Class that allows to create a SQL data base for a PDB file
This allows to easily extract information of the PDB using SQL queries

USAGE db = pdb2sql('XXX.pdb')

A few SQL querry wrappers have been implemented 

	
	self.get(attribute_name,**kwargs)
	
		Get hte value(s) of the attribute(s) for possible selection of the db

		attributename   : 	must be a valid attribute name. 
						 	you can get these names via the get_colnames()
						 	serial, name, atLoc,resName,chainID, resSeq,iCode,x,y,z,occ,temp
						 	you can specify more than one attribute name at once e.g 'x,y,z'

		keyword args    :   Several options are possible
							None : return the entire table

							chain = 'X' return the values of that chain
							name  = 'CA' only these atoms
							index = [0,1,2,3] return only those rows (not serial) 
							where = "chainID='B'" general WHERE SQL query 
							query = 'WHERE chainID='B'' general SQL Query

		example         :

							db = pdb2sql(filename)
							xyz  = db.get('x,y,z',index=[0,1,2,3])
							name = db.get('name',where="resName='VAL'")

	self.put(attribute_name,value,**kwargs)

		Update the value of the attribute with value specified with possible selection

		attributename   : 	must be a valid attribute name. 
						 	you can get these names via the get_colnames()
						 	serial, name, atLoc,resName,chainID, resSeq,iCode,x,y,z,occ,temp
						 	you can specify more than one attribute name at once e.g 'x,y,z'

		keyword args    :   Several options are possible
							None : put the value in the entire column

							index = [0,1,2,3] in only these indexes (not serial)
							where = "chainID='B'" only for this chain
							query = general SQL Query

		example         :

							db = pdb2sql(filename)
							db.add_column('CHARGE')
							db.put('CHARGE',1.25,index=[1])							
							db.close()


	Other queries have been made user friendly

	- self.add_column(column_name,coltype='FLOAT',default=0)
	- self.update_column(colname,values,index=None)
	- self.update_xyz(new_xyz,index=None)
	- self.commit()

	TO DO 

	- Add more user friendly wrappers to SQL queries
	- Make use of the ? more often to prevent quoting issues and SQL injection attack 

'''

class pdb2sql(object):

	'''
	CLASS that transsform  PDB file into a sqlite database
	'''

	def __init__(self,pdbfile,sqlfile=None,fix_chainID=False,verbose=False):

		self.pdbfile = pdbfile
		self.sqlfile = sqlfile
		self.is_valid = True
		self.verbose = verbose

		# create the database
		self._create_sql()


		# fix the chain ID
		if fix_chainID:
			self._fix_chainID()

		# backbone type
		self.backbone_type = ['C','CA','N','O']

	##################################################################################
	#
	#	CREATION AND PRINTING
	#
	##################################################################################

	'''
	Main function to create the SQL data base
	'''
	def _create_sql(self):

		pdbfile = self.pdbfile
		sqlfile = self.sqlfile

		if self.verbose:
			print('-- Create SQLite3 database')

		 #name of the table
		table = 'ATOM'

		# column names and types
		self.col = {'serial' : 'INT',
		       'name'   : 'TEXT',
		       'altLoc' : 'TEXT',
		       'resName' : 'TEXT',
		       'chainID' : 'TEXT',
			   'resSeq'  : 'INT',
			   'iCode'   : 'TEXT',
			   'x'       : 'REAL',
			   'y'       : 'REAL',
			   'z'       : 'REAL',
			   'occ'     : 'REAL',
			   'temp'    : 'REAL'}

	    # delimtier of the column format
	    # taken from http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
		self.delimiter = {
					'serial' : [6,11],
					'name'   : [12,16],
					'altLoc' : [16,17],
					'resName' :[17,20],
					'chainID' :[21,22],
					'resSeq'  :[22,26],
					'iCode'   :[26,26],
					'x'       :[30,38],
					'y'       :[38,46],
					'z'       :[46,54],
					'occ'     :[54,60],
					'temp'    :[60,66]}
	    
	    # size of the things
		ncol = len(self.col)
		ndel = len(self.delimiter)


	    # open the data base 
	    # if we do not specify a db name 
	    # the db is only in RAM
	    # there might be little advantage to use memory
	    # https://stackoverflow.com/questions/764710/sqlite-performance-benchmark-why-is-memory-so-slow-only-1-5x-as-fast-as-d
		if self.sqlfile is None:
			self.conn = sqlite3.connect(':memory:')
		# or we create a new db file
		else:
			if os.path.isfile(sqlfile):
				sp.call('rm %s' %sqlfile,shell=True)
			self.conn = sqlite3.connect(sqlfile)
		self.c = self.conn.cursor()

		# intialize the header/placeholder
		header,qm = '',''
		for ic,(colname,coltype) in enumerate(self.col.items()):
			header += '{cn} {ct}'.format(cn=colname,ct=coltype)
			qm += '?'
			if ic < ncol-1:
				header += ', '
				qm += ','

		# create the table
		query = 'CREATE TABLE ATOM ({hd})'.format(hd=header)
		self.c.execute(query)
		

		# read the pdb file
		# this is dangerous if there are ATOM written in the comment part 
		# which happends often
		#data = sp.check_output("awk '/ATOM/' %s" %pdbfile,shell=True).decode('utf8').split('\n')

		# a safer version consist at matching against the first field
		# won't work on windows
		#data = sp.check_output("awk '$1 ~ /^ATOM/' %s" %pdbfile,shell=True).decode('utf8').split('\n')

		# a pure python way
		# RMK we go through the data twice here. Once to read the ATOM line and once to parse the data ... 
		# we could do better than that. But the most time consuming step seems to be the CREATE TABLE query
		# if we path a file we read it
		if isinstance(pdbfile,str):
			if os.path.isfile(pdbfile):
				with open(pdbfile,'r') as fi:
					data = [line.split('\n')[0] for line in fi if line.startswith('ATOM')]

		# if we pass a list as for h5py read/write
		# we directly use that
		elif isinstance(pdbfile,np.ndarray):
			data =  [l.decode('utf-8') for l in pdbfile.tolist()]

		# if we cant read it
		else:
			print(pdbfile)
			raise ValueError('PDB data not recognized')

		# if there is no ATOM in the file
		if len(data)==1 and data[0]=='':
			print("-- Error : No ATOM in the pdb file.")
			self.is_valid = False
			return

		# haddock chain ID fix
		del_copy = self.delimiter.copy()
		if data[0][del_copy['chainID'][0]] == ' ':
			del_copy['chainID'] = [72,73]

		# get all the data
		data_atom = []
		for iatom,atom in enumerate(data):

			# sometimes we still have an empty line somewhere
			if len(atom) == 0:
				continue

			# browse all attribute of each atom
			at = ()
			for ik,(colname,coltype) in enumerate(self.col.items()):

				# get the piece of data
				data = atom[del_copy[colname][0]:del_copy[colname][1]].strip()

				# convert it if necessary
				if coltype == 'INT':
					data = int(data)
				elif coltype == 'REAL':
					data = float(data)

				# append keep the comma !!
				# we need proper tuple
				at +=(data,)

			# append
			data_atom.append(at)


		# push in the database
		self.c.executemany('INSERT INTO ATOM VALUES ({qm})'.format(qm=qm),data_atom)
	

	# replace the chain ID by A,B,C,D, ..... in that order
	def _fix_chainID(self):

		from string import ascii_uppercase 

		# get the current names
		chainID = self.get('chainID')
		natom = len(chainID)
		chainID = sorted(set(chainID))

		if len(chainID)>26:
			print("Warning more than 26 chains have been detected. This is so far not supported")
			sys.exit()

		# declare the new names
		newID = [''] * natom

		# fill in the new names
		for ic,chain in enumerate(chainID):
			index = self.get('rowID',chainID=chain)
			for ind in index:
				newID[ind] = ascii_uppercase[ic]

		# update the new name
		self.update_column('chainID',newID)


	# get the names of the columns
	def get_colnames(self):
		cd = self.conn.execute('select * from atom')
		print('Possible column names are:')
		names = list(map(lambda x: x[0], cd.description))
		print('\trowID')
		for n in names:
			print('\t'+n)

	# print the database
	def prettyprint(self):
		import pandas.io.sql as psql 
		df = psql.read_sql("SELECT * FROM ATOM",self.conn)
		print(df)

	def uglyprint(self):
		ctmp = self.conn.cursor()
		ctmp.execute("SELECT * FROM ATOM")
		print(ctmp.fetchall())


	############################################################################################
	#
	#		GET FUNCTIONS
	#
	#			get(attribute,selection) -> return the atribute(s) value(s) for the given selection 
	#			get_contact_atoms()		 -> return a list of rowID  for the contact atoms
	#			get_contact_residue()	 -> return a list of resSeq for the contact residue
	#
	###############################################################################################

	# get the properties
	def get(self,atnames,**kwargs):

		'''
		Exectute a simple SQL query that extracts values of attributes for certain condition
		Ex  db.get('x,y,z',chainID='A', name = ['C','CA'])
		returns an array containing the value of the attributes
		'''

		# the asked keys
		keys = kwargs.keys()			

		# check if the column exists
		try:
			self.c.execute("SELECT EXISTS(SELECT {an} FROM ATOM)".format(an=atnames))
		except:
			print('Error column %s not found in the database' %atnames)
			self.get_colnames()
			return

		# if we have 0 key we take the entire db
		if len(kwargs) == 0:
			query = 'SELECT {an} FROM ATOM'.format(an=atnames)
			data = [list(row) for row in self.c.execute(query)]
		
		############################################################################
		# GENERIC QUERY
		#
		# the only one we need
		# each keys must be a valid columns
		# each value may be a single value or an array
		# AND is assumed between different keys
		# OR is assumed for the different values of a given key
		#
		##############################################################################
		else:

			# check that all the keys exists
			for k in keys:
				try:
					self.c.execute("SELECT EXISTS(SELECT {an} FROM ATOM)".format(an=k))
				except:
					print('Error column %s not found in the database' %k)
					self.get_colnames()
					return

			# form the query and the tuple value
			query = 'SELECT {an} FROM ATOM WHERE '.format(an=atnames)
			conditions = []
			vals = ()

			# iterate through the kwargs
			for ik,(k,v) in enumerate(kwargs.items()):

				# get if we have an array or a scalar
				# and build the value tuple for the sql query
				# deal with the indexing issue if rowID is required
				if isinstance(v,list):
					nv = len(v)
					if k == 'rowID':
						vals = vals + tuple([iv+1 for iv in v])
					else:
						vals = vals + tuple(v)
				else:
					nv = 1
					if k == 'rowID':
						vals = vals + (v+1,)
					else:
						vals = vals + (v,)

				# create the condition for that key
				conditions.append(k + ' in (' + ','.join('?'*nv) + ')')

			# stitch the conditions and append to the query
			query += ' AND '.join(conditions)	

			# query the sql database and return the answer in a list
			data = [list(row) for row in self.c.execute(query,vals)]
		
		# empty data
		if len(data)==0:
			print('Warning sqldb.get returned an empty')
			return data

		# fix the python <--> sql indexes
		# if atnames == 'rowID':
		if 'rowID' in atnames:
			index = atnames.split(',').index('rowID')
			for i in range(len(data)):
				data[i][index] -= 1

		# postporcess the output of the SQl query
		# flatten it if each els is of size 1
		if len(data[0])==1:
			data = [d[0] for d in data]
	
		return data

	############################################################################
	#
	# get the contact atoms
	# 
	# we should have a entire module called pdb2sql
	# with a submodule pdb2sql.interface that finds contact atoms/residues
	# and possbily other submodules to do other things
	# that will leave only the get / put methods in the main class
	#
	#############################################################################
	def get_contact_atoms(self,cutoff=8.5,chain1='A',chain2='B',
		                  extend_to_residue=False,only_backbone_atoms=False,
		                  excludeH=False,return_only_backbone_atoms=False,return_contact_pairs=False):

		# xyz of the chains
		xyz1 = np.array(self.get('x,y,z',chainID=chain1))
		xyz2 = np.array(self.get('x,y,z',chainID=chain2))

		# index of b
		index2 = self.get('rowID',chainID=chain2)
		
		# resName of the chains
		resName1 = np.array(self.get('resName',chainID=chain1))
		resName2 = np.array(self.get('resName',chainID=chain2))

		# atomnames of the chains
		atName1 = np.array(self.get('name',chainID=chain1))
		atName2 = np.array(self.get('name',chainID=chain2))


		# loop through the first chain
		# TO DO : loop through the smallest chain instead ... 
		index_contact_1,index_contact_2 = [],[]
		index_contact_pairs = {}

		for i,x0 in enumerate(xyz1):

			# compute the contact atoms
			contacts = np.where(np.sqrt(np.sum((xyz2-x0)**2,1)) <= cutoff )[0]

			# exclude the H if required
			if excludeH and atName1[i][0] == 'H':
				continue

			if len(contacts)>0 and any([not only_backbone_atoms, atName1[i] in self.backbone_type]):

				# the contact atoms
				index_contact_1 += [i]
				index_contact_2 += [index2[k] for k in contacts if ( any( [atName2[k] in self.backbone_type,  not only_backbone_atoms]) and not (excludeH and atName2[k][0]=='H') ) ]
				
				# the pairs
				pairs = [index2[k] for k in contacts if any( [atName2[k] in self.backbone_type,  not only_backbone_atoms] ) and not (excludeH and atName2[k][0]=='H') ]
				if len(pairs) > 0:
					index_contact_pairs[i] = pairs

		# get uniques
		index_contact_1 = sorted(set(index_contact_1))
		index_contact_2 = sorted(set(index_contact_2))

		# if no atoms were found	
		if len(index_contact_1)==0:
			print('Warning : No contact atoms detected in pdb2sql')

		# extend the list to entire residue
		if extend_to_residue:
			index_contact_1,index_contact_2 = self._extend_contact_to_residue(index_contact_1,index_contact_2,only_backbone_atoms)	


		# filter only the backbone atoms
		if return_only_backbone_atoms and not only_backbone_atoms:

			# get all the names
			# there are better ways to do that !
			atNames = np.array(self.get('name'))

			# change the index_contacts
			index_contact_1 = [  ind for ind in index_contact_1 if atNames[ind] in self.backbone_type ]
			index_contact_2 = [  ind for ind in index_contact_2 if atNames[ind] in self.backbone_type ]

			# change the contact pairs
			tmp_dict = {}
			for ind1,ind2_list in index_contact_pairs.items():

				if atNames[ind1] in self.backbone_type:
					tmp_dict[ind1] = [ind2 for ind2 in ind2_list if atNames[ind2] in self.backbone_type]

			index_contact_pairs = tmp_dict

		# not sure that's the best way of dealing with that
		if return_contact_pairs:
			return index_contact_pairs
		else:
			return index_contact_1,index_contact_2

	# extend the contact atoms to the residue
	def _extend_contact_to_residue(self,index1,index2,only_backbone_atoms):

		# extract the data
		dataA = self.get('chainId,resName,resSeq',rowID=index1)
		dataB = self.get('chainId,resName,resSeq',rowID=index2)

		# create tuple cause we want to hash through it
		dataA = list(map(lambda x: tuple(x),dataA))
		dataB = list(map(lambda x: tuple(x),dataB))

		# extract uniques
		resA = list(set(dataA))
		resB = list(set(dataB))

		# init the list
		index_contact_A,index_contact_B = [],[]

		# contact of chain A
		for resdata in resA:
			chainID,resName,resSeq = resdata
			
			if only_backbone_atoms:
				index_contact_A += self.get('rowID',chainID=chainID,resName=resName,resSeq=resSeq,name=self.backbone_type)
			else:
				index_contact_A += self.get('rowID',chainID=chainID,resName=resName,resSeq=resSeq)
		
		# contact of chain B
		for resdata in resB:
			chainID,resName,resSeq = resdata
			
			if only_backbone_atoms:
				index_contact_B += self.get('rowID',chainID=chainID,resName=resName,resSeq=resSeq,name=self.backbone_type)
			else:
				index_contact_B += self.get('rowID',chainID=chainID,resName=resName,resSeq=resSeq)

		# make sure that we don't have double (maybe optional)
		index_contact_A = sorted(set(index_contact_A))
		index_contact_B = sorted(set(index_contact_B))
		
		return index_contact_A,index_contact_B		


	# get the contact residue
	def get_contact_residue(self,cutoff=8.5,chain1='A',chain2='B',excludeH=False,
		                    only_backbone_atoms=False,return_contact_pairs=False):

		# get the contact atoms
		if return_contact_pairs:

			# declare the dict
			residue_contact_pairs = {}

			# get the contact atom pairs
			atom_pairs = self.get_contact_atoms(cutoff=cutoff,chain1=chain1,chain2=chain2,
				                                only_backbone_atoms=only_backbone_atoms,
				                                excludeH=excludeH,
				                                return_contact_pairs=True)

			# loop over the atom pair dict
			for iat1,atoms2 in atom_pairs.items():

				# get the res info of the current atom
				data1 = tuple(self.get('chainID,resSeq,resName',rowID=[iat1])[0])

				# create a new entry in the dict if necessary
				if data1 not in residue_contact_pairs:
					residue_contact_pairs[data1] = set()

				# get the res info of the atom in the other chain
				data2 = self.get('chainID,resSeq,resName',rowID=atoms2)

				# store that in the dict without double
				for resData in data2:
					residue_contact_pairs[data1].add(tuple(resData))

			for resData in residue_contact_pairs.keys():
				residue_contact_pairs[resData] = sorted(residue_contact_pairs[resData])

			return residue_contact_pairs

		else:

			# get the contact atoms
			contact_atoms = self.get_contact_atoms(cutoff=cutoff,chain1=chain1,chain2=chain2,return_contact_pairs=False)

			# get the residue info
			data1 = self.get('chainID,resSeq,resName',rowID=contact_atoms[0])
			data2 = self.get('chainID,resSeq,resName',rowID=contact_atoms[1])

			# take only unique
			residue_contact_A = sorted(set([tuple(resData) for resData in data1]))
			residue_contact_B = sorted(set([tuple(resData) for resData in data2]))

			return residue_contact_A,residue_contact_B


	############################################################################################
	#
	#		PUT FUNCTONS AND ASSOCIATED
	#
	#			add_column()	-> add a column
	#			update_column() -> update the values of one column
	#			update_xyz()    -> update_xyz of the pdb
	#			put()           -> put a value in a column
	#
	###############################################################################################


	def add_column(self,colname,coltype='FLOAT',default=0):

		'''
		Add an etra column to the ATOM table
		'''
		query = "ALTER TABLE ATOM ADD COLUMN '%s' %s DEFAULT %s" %(colname,coltype,str(default))
		self.c.execute(query)
		#self.conn.commit()

	def update(self,attribute,values,**kwargs):

		# the asked keys
		keys = kwargs.keys()			

		# check if the column exists
		try:
			self.c.execute("SELECT EXISTS(SELECT {an} FROM ATOM)".format(an=attribute))
		except:
			print('Error column %s not found in the database' %attribute)
			self.get_colnames()
			return

		# handle the multi model cases 
		if 'model' not in keys and self.nModel > 0:
			for iModel in range(self.nModel):
				kwargs['model'] = iModel
				self.update(attribute,values,**kwargs)
			return 

		# parse the attribute
		if ',' in attribute:
			attribute = attribute.split(',')

		if not isinstance(attribute,list):
			attribute = [attribute]


		# check the size
		natt = len(attribute)
		nrow = len(values)
		ncol = len(values[0])

		if natt != ncol:
			raise ValueError('Number of attribute incompatible with the number of columns in the data')



		# get the row ID of the selection
		rowID = self.get('rowID',**kwargs)
		nselect = len(rowID)

		if nselect != nrow:
			raise ValueError('Number of data values incompatible with the given conditions')

		# prepare the query
		query = 'UPDATE ATOM SET '
		query = query + ', '.join(map(lambda x: x+'=?',attribute))
		if len(kwargs)>0:
			query = query + ' WHERE rowID=?'
			

		# prepare the data
		data = []
		for i,val in enumerate(values):

			tmp_data = [ v for v in val ]

			if len(kwargs)>0:

				# here the conversion of the indexes is a bit annoying
				tmp_data += [rowID[i]+1]

			data.append(tmp_data)

		self.c.executemany(query,data)

	def update_column(self,colname,values,index=None):


		'''
		values must contain the correct number of elements
		'''

		if index==None:
			data = [ [v,i+1] for i,v in enumerate(values) ]
		else:
			data = [ [v,ind] for v,ind in zip(values,index)] # shouldn't that be ind+1 ?

		query = 'UPDATE ATOM SET {cn}=? WHERE rowID=?'.format(cn=colname)
		self.c.executemany(query,data)
		#self.conn.commit()

	def update_xyz(self,xyz,index=None):

		'''
		update the positions of the atoms selected
		if index=None the position of all the atoms are changed
		'''

		if index==None:
			data = [ [pos[0],pos[1],pos[2],i+1] for i,pos in enumerate(xyz) ]
		else:
			data = [ [pos[0],pos[1],pos[2],ind+1] for pos,ind in zip(xyz,index)]

		query = 'UPDATE ATOM SET x=?, y=?, z=? WHERE rowID=?'
		self.c.executemany(query,data)

	def put(self,colname,value,**kwargs):

		'''
		Exectute a simple SQL query that put a value in an attributes for certain condition
		Ex  db.put('resName','XXX',where="chainID=='A'")
		'''
		
		arguments = {'where' : "String e.g 'chainID = 'A''",
					 'index' : "Array e.g. [27,28,30]",
					 'name'  : "'CA' atome name",
					 'query' : "SQL query e.g. 'WHERE chainID='B' AND resName='ASP' "}

		# the asked keys
		keys = kwargs.keys()			

		# if we have more than one key we kill it
		if len(keys)>1 :
			print('You can only specify 1 conditional statement for the pdb2sql.put function')
			return 

		# check if the column exists
		try:
			self.c.execute("SELECT EXISTS(SELECT {an} FROM ATOM)".format(an=colname))
		except:
			print('Error column %s not found in the database' %colname)
			self.get_colnames()
			return


		# if we have 0 key we take the entire db
		if len(kwargs) == 0:
			query = 'UPDATE ATOM SET {cn}=?'.format(cn=colname)
			value = tuple([value])
			self.c.execute(query,value)
			return  

		# otherwise we have only one key
		key = list(keys)[0]
		cond = kwargs[key]

		# select which key we have
		if key == 'where':
			query = 'UPDATE ATOM SET {cn}=? WHERE {cond}'.format(cn=colname,cond=cond)
			value = tuple([value])
			self.c.execute(query,value)

		elif key == 'name' :
			values = tuple([value,cond])
			query = 'UPDATE ATOM SET {cn}=? WHERE name=?'.format(cn=colname)
			self.c.execute(query,values)

		elif key == 'index' :
			values = tuple([value] + [v+1 for v in cond])
			qm = ','.join(['?' for i in range(len(cond))])
			query = 'UPDATE ATOM SET {cn}=? WHERE rowID in ({qm})'.format(cn=colname,qm=qm)
			self.c.execute(query,values)
		
		elif key == 'query' :
			query = 'UPDATE ATOM SET {cn}=? {c1}'.format(cn=colname,c1=cond)
			value = tuple([value])
			self.c.execute(query,value)

		else:
			print('Error arguments %s not supported in pdb2sql.get()\nOptions are:\n' %(key))
			for posskey,possvalue in arguments.items():
				print('\t' + posskey + '\t\t' + possvalue)
			return



	############################################################################################
	#
	#		COMMIT, EXPORT, CLOSE FUNCTIONS
	#
	###############################################################################################

	# comit changes 
	def commit(self):
		self.conn.commit()

	# export to pdb file
	def exportpdb(self,fname,**kwargs):

		'''
		Export a PDB file with kwargs selection
		not pretty so far but functional
		'''

		# get the data
		data = self.get('*',**kwargs)

		# write each line
		# the PDB format is pretty strict
		# http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
		f = open(fname,'w')
		for d in data:
			line = 'ATOM  '
			line += '{:>5}'.format(d[0])	# serial
			line += ' '
			line += '{:^4}'.format(d[1])	# name
			line += '{:>1}'.format(d[2])	# altLoc
			line += '{:>3}'.format(d[3])	#resname
			line += ' '
			line += '{:>1}'.format(d[4])	# chainID
			line += '{:>4}'.format(d[5])	# resSeq
			line += '{:>1}'.format(d[6])	# iCODE
			line += '   '
			line += '{: 8.3f}'.format(d[7])	#x
			line += '{: 8.3f}'.format(d[8])	#y
			line += '{: 8.3f}'.format(d[9])	#z
			line += '{: 6.2f}'.format(d[10])	# occ
			line += '{: 6.2f}'.format(d[11])	# temp
			line += '\n'

			f.write(line)

		# close
		f.close()


	# close the database 
	def close(self,rmdb = True):
		
		if self.sqlfile is None:
			self.conn.close()

		else:

			if rmdb:
				self.conn.close() 
				os.system('rm %s' %(self.sqlfile))
			else:
				self.commit()
				self.conn.close() 

	############################################################################
	#
	# Transform the position of the molecule
	#
	##############################################################################


	def translation(self,vect,**kwargs):
		xyz = self.get('x,y,z',**kwargs)
		xyz += vect
		self.update('x,y,z',xyz,**kwargs)

	def rotation_around_axis(self,axis,angle,**kwargs):

		xyz = self.get('x,y,z',**kwargs)

		# get the data
		ct,st = np.cos(angle),np.sin(angle)
		ux,uy,uz = axis

		# get the center of the molecule
		xyz0 = np.mean(xyz,0)

		# definition of the rotation matrix
		# see https://en.wikipedia.org/wiki/Rotation_matrix
		rot_mat = np.array([
		[ct + ux**2*(1-ct),			ux*uy*(1-ct) - uz*st,		ux*uz*(1-ct) + uy*st],
		[uy*ux*(1-ct) + uz*st,    	ct + uy**2*(1-ct),			uy*uz*(1-ct) - ux*st],
		[uz*ux*(1-ct) - uy*st,		uz*uy*(1-ct) + ux*st,   	ct + uz**2*(1-ct)   ]])

		# apply the rotation
		xyz = np.dot(rot_mat,(xyz-xyz0).T).T + xyz0

		self.update('x,y,z',xyz,**kwargs)

		return xyz0
			
	def rotation_euler(self,alpha,beta,gamma,**kwargs):

		xyz = self.get('x,y,z',**kwargs)

		# precomte the trig
		ca,sa = np.cos(alpha),np.sin(alpha)
		cb,sb = np.cos(beta),np.sin(beta)
		cg,sg = np.cos(gamma),np.sin(gamma)


		# get the center of the molecule
		xyz0 = np.mean(xyz,0)

		# rotation matrices
		rx = np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
		ry = np.array([[cb,0,sb],[0,1,0],[-sb,0,cb]])
		rz = np.array([[cg,-sg,0],[sg,cs,0],[0,0,1]])

		rot_mat = np.dot(rz,np.dot(ry,rz))

		# apply the rotation
		xyz = np.dot(rot_mat,(xyz-xyz0).T).T + xyz0

		self.update('x,y,z',xyz,**kwargs)

	def rotation_matrix(self,rot_mat,center=True,**kwargs):

		xyz = self.get('x,y,z',**kwargs)

		if center:
			xyz0 = np.mean(xyz)
			xyz = np.dot(rot_mat,(xyz-xyz0).T).T + xyz0
		else:
			xyz = np.dot(rot_mat,(xyz).T).T
		self.update('x,y,z',xyz,**kwargs)









if __name__ == '__main__':

	import numpy as np

	# create the sql
	db = pdb2sql('1AK4_100w.pdb')

	# print the database
	db.prettyprint()

	# get the names of the columns
	db.get_colnames()

	# extract the xyz position of the atoms with name CB
	xyz = db.get('*',index=[0,1,2,3])
	print(xyz)

	xyz = db.get('rowID',where="resName='VAL'")
	print(xyz)

	db.add_column('CHARGE','FLOAT')
	db.put('CHARGE',0.1)
	db.prettyprint()

	db.exportpdb('chainA.pdb',where="chainID='A'")

	# close the database
	db.close()