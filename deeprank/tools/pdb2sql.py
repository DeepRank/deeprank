import sqlite3
import subprocess as sp 
import os
import numpy as np

'''
Class that allows to create a SL data base for a PDB file
This allows to easily extract information of the PDB using SQL query

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

	def __init__(self,pdbfile,sqlfile='pdb2sql.db',fix_chainID=True,verbose=False):

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

		# remove the database if necessary
		if os.path.isfile(sqlfile):
			sp.call('rm %s' %sqlfile,shell=True)

	    # open the data base
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
		data = sp.check_output("awk '$1 ~ /^ATOM/' %s" %pdbfile,shell=True).decode('utf8').split('\n')

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

		# commit the change
		self.conn.commit()


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
			index = self.get('rowID',chain=chain)
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

	# get the properties
	def get(self,atnames,**kwargs):

		'''
		Exectute a simple SQL query that extracts values of attributes for certain condition
		Ex  db.get('x,y,z',where="chainIN=='A'")
		returns an array containing the value of the attributes
		'''
		
		arguments = {'where' : "String e.g 'chainID = 'A''",
					 'index' : "Array e.g. [27,28,30]",
					 'chain' : "Char e.g. 'A'",
					 'name'  : "String e.g 'CA'",
					 'query' : "SQL query e.g. 'WHERE chainID='B''"}

		# the asked keys
		keys = kwargs.keys()			

		# if we have more than one key we kill it
		if len(keys)>1 :
			print('Error :You can only specify 1 conditional statement for the pdb2sql.get function')
			print('For complex query use the query kw-argument of pdb2sql.get()')
			print("Example : sqldb.get('x,y,z',query='WHERE chainID='A' AND resName='VAL'")
			return 

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
			

		# otherwise we have only one key
		else:

			key = list(keys)[0]
			value = kwargs[key]

			# select which key we have
			if key == 'where':
				query =  'SELECT {an} FROM ATOM WHERE {c1}'.format(an=atnames,c1=value)
				data = [list(row) for row in self.c.execute(query)]

			elif key == 'index' :

				# we can have a max of 999 value in a SQL QUERY
				# for larger index range we need to proceed by batch

				SQL_LIMIT = 999
				nbatch = int(len(value)/999)+1
				data = []
				for ibatch in range(nbatch):

					start,end = ibatch*SQL_LIMIT,(ibatch+1)*SQL_LIMIT

					value_tmp = tuple([v+1 for v in value[start:end]])
					qm = ','.join(['?' for i in range(len(value_tmp))])

					query  =  'SELECT {an} FROM ATOM WHERE rowID in ({qm})'.format(an=atnames,qm=qm)
					data += [list(row) for row in self.c.execute(query,value_tmp)]

			elif key == 'chain':
				query = "SELECT {an} FROM ATOM WHERE chainID=?".format(an=atnames)
				data = [list(row) for row in self.c.execute(query,value)]

			elif key == 'name':
				query = "SELECT {an} FROM ATOM WHERE name='{name}'".format(an=atnames,name=value)
				data = [list(row) for row in self.c.execute(query)]
			
			elif key == 'query' :	
				query = 'SELECT {an} FROM ATOM {c1}'.format(an=atnames,c1=value)
				data = [list(row) for row in self.c.execute(query)]

			else:
				print('Error arguments %s not supported in pdb2sql.get()\nOptions are:\n' %(key))
				for posskey,possvalue in arguments.items():
					print('\t' + posskey + '\t\t' + possvalue)
				return
		
		# empty data
		if len(data)==0:
			print('Warning sqldb.get returned an empty')
			return data

		# postporcess the output of the SQl query
		# flatten it if each els is of size 1
		if len(data[0])==1:
			data = [d[0] for d in data]

		# fix the python <--> sql indexes
		if atnames == 'rowID':
			data = [d-1 for d in data]
			#data = np.sort(np.array(data)).tolist()
		
		return data

	# not entirely sure but I think that's useless
	def get_indexes(self,atnames,index):
		strind = ','.join(map(str,index))
		return self.get(atnames,where="rowID in ({ind})".format(ind=strind))

	# get the contact atoms
	def get_contact_atoms(self,cutoff=8.5,chain1='A',chain2='B',
		                  extend_to_residue=False,only_backbone_atoms=False,return_contact_pairs=False):

		# xyz of the chains
		xyz1 = np.array(self.get('x,y,z',chain=chain1))
		xyz2 = np.array(self.get('x,y,z',chain=chain2))

		# index of b
		index2 = self.get('rowID',chain=chain2)
		
		# resName of the chains
		resName1 = np.array(self.get('resName',chain=chain1))
		resName2 = np.array(self.get('resName',chain=chain2))

		# atomnames of the chains
		atName1 = np.array(self.get('name',chain=chain1))
		atName2 = np.array(self.get('name',chain=chain2))


		# loop through the first chain
		# TO DO : loop through the smallest chain instead ... 
		index_contact_1,index_contact_2 = [],[]
		index_contact_pairs = {}

		for i,x0 in enumerate(xyz1):

			# compute the contact atoms
			contacts = np.where(np.sqrt(np.sum((xyz2-x0)**2,1)) < cutoff )[0]

			if len(contacts)>0 and any([not only_backbone_atoms, atName1[i] in self.backbone_type]):

				index_contact_1 += [i]
				index_contact_2 += [index2[k] for k in contacts if any( [atName2[k] in self.backbone_type,  not only_backbone_atoms] )]
				index_contact_pairs[i] = [index2[k] for k in contacts if any( [atName2[k] in self.backbone_type,  not only_backbone_atoms] )]

		# get uniques
		index_contact_1 = sorted(set(index_contact_1))
		index_contact_2 = sorted(set(index_contact_2))

		# if no atoms were found	
		if len(index_contact_1)==0:
			print('Warning : No contact atoms detected in pdb2sql')

		if extend_to_residue:
			index_contact_1,index_contact_2 = self._extend_contact_to_residue(index_contact_1,index_contact_2,only_backbone_atoms)	

		# not sure that's the best way of dealing with that
		if return_contact_pairs:
			return index_contact_pairs
		else:
			return index_contact_1,index_contact_2

	# extend the contact atoms to the residue
	def _extend_contact_to_residue(self,index1,index2,only_backbone_atoms):

		# extract the data
		dataA = self.get('chainId,resName,resSeq',index=index1)
		dataB = self.get('chainId,resName,resSeq',index=index2)

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
			query = "WHERE chainID='{chainID}' AND resName='{resName}' AND resSeq={resSeq}".format(chainID=chainID,resName=resName,resSeq=resSeq)
			if only_backbone_atoms:
				index = self.get('rowID',query=query)
				name = self.get('name',query=query)
				index_contact_A += [ ind for ind,n in zip(index,name) if n in self.backbone_type ]
			else:
				index_contact_A += self.get('rowID',query=query)
		
		# contact of chain B
		for resdata in resB:
			chainID,resName,resSeq = resdata
			query = "WHERE chainID='{chainID}' AND resName='{resName}' AND resSeq={resSeq}".format(chainID=chainID,resName=resName,resSeq=resSeq)
			if only_backbone_atoms:
				index = self.get('rowID',query=query)
				name = self.get('name',query=query)
				index_contact_B += [ ind for ind,n in zip(index,name) if n in self.backbone_type ]
			else:
				index_contact_B += self.get('rowID',query=query)

		# make sure that we don't have double (maybe optional)
		index_contact_A = sorted(set(index_contact_A))
		index_contact_B = sorted(set(index_contact_B))
		
		return index_contact_A,index_contact_B		

	def add_column(self,colname,coltype='FLOAT',default=0):

		'''
		Add an etra column to the ATOM table
		'''
		query = "ALTER TABLE ATOM ADD COLUMN '%s' %s DEFAULT %s" %(colname,coltype,str(default))
		self.c.execute(query)
		self.conn.commit()

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


	def commit(self):
		self.conn.commit()


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
		self.conn.close() 
		if rmdb:
			os.system('rm %s' %(self.sqlfile))


















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