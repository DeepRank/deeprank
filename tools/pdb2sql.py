import sqlite3
import subprocess as sp 
import os

'''
Class that allows to create a SL data base for a PDB file
This allows to easily extract information of the PDB using SQL query

USAGE db = pdb2sql('XXX.pdb')

A main SQL querry handler has been implemented (might not be the best idea)

	pdb2sql.get(attribute_name,**kwargs)
	
	attributename   : must be a valid attribute name. 
                      you can get these names via the get_colnames()
                      serial, name, atLoc,resName,chainID, resSeq,iCode,x,y,z,occ,temp
                      you can specify more than one attribute name at once e.g 'x,y,z'

	keyword args    : Several options are possible
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
'''

class pdb2sql(object):

	'''
	CLASS that transsform  PDB file into a sqlite database
	'''

	def __init__(self,pdbfile,sqlfile='pdb2sql.db'):
		self.pdbfile = pdbfile
		self.sqlfile = sqlfile
		self._create_sql()

	'''
	Main function to create the SQL data base
	'''
	def _create_sql(self):

		pdbfile = self.pdbfile
		sqlfile = self.sqlfile

		print('CREATE SQLite DATABASE FOR FILE %s at %s' %(pdbfile,sqlfile))

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
		data = sp.check_output("awk '/ATOM/' %s" %pdbfile,shell=True).decode('utf8').split('\n')

		# hddock chain ID fix
		del_copy = self.delimiter.copy()
		if data[0][del_copy['chainID'][0]] == ' ':
			print('Deprecated FORMAT DETECTED CHAINID AT THE END')
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

				# append
				at +=(data,)

			# append
			data_atom.append(at)

		# push in the database
		self.c.executemany('INSERT INTO ATOM VALUES ({qm})'.format(qm=qm),data_atom)

		# commit the change
		self.conn.commit()


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
			print('You can only specify 1 conditional statement for the pdb2sql.get function')
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
			data = [list(row) for row in self.c.execute('SELECT {an} FROM ATOM'.format(an=atnames))]
			return data 

		# otherwise we have only one key
		key = list(keys)[0]
		value = kwargs[key]

		# select which key we have
		if key == 'where':
			data = [list(row) for row in self.c.execute('SELECT {an} FROM ATOM WHERE {c1}'.format(an=atnames,c1=value))]

		elif key == 'index' :
			value = [v+1 for v in value]
			strind = ','.join(map(str,value))
			data = [list(row) for row in self.c.execute('SELECT {an} FROM ATOM WHERE rowID in ({c1})'.format(an=atnames,c1=strind))]

		elif key == 'chain':	
			data = [list(row) for row in self.c.execute("SELECT {an} FROM ATOM WHERE chainID= '{c1}'".format(an=atnames,c1=value))]

		elif key == 'name':
			data = [list(row) for row in self.c.execute("SELECT {an} FROM ATOM WHERE name= '{c1}'".format(an=atnames,c1=value))]
		
		elif key == 'query' :	
				data = [list(row) for row in self.c.execute('SELECT {an} FROM ATOM {c1}'.format(an=atnames,ind=value))]

		else:
			print('Error arguments %s not supported in pdb2sql.get()\nOptions are:\n' %(key))
			for posskey,possvalue in arguments.items():
				print('\t' + posskey + '\t\t' + possvalue)
			return
		

		# postporcess the output of the SQl query
		# flatten it if each els is of size 1
		if len(data[0])==1:
			data = [d[0] for d in data]

		# fix the python <--> sql indexes
		if atnames == 'rowID':
			data = [d-1 for d in data]
		
		return data

	def get_indexes(self,atnames,index):
		strind = ','.join(map(str,index))
		return self.get(atnames,where="rowID in ({ind})".format(ind=strind))


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
	#db.prettyprint()

	# get the names of the columns
	#db.get_colnames()

	# extract the xyz position of the atoms with name CB
	xyz = db.get('*',index=[0,1,2,3])
	print(xyz)

	xyz = db.get('rowID',where="resName='VAL'")
	print(xyz)

	db.exportpdb('chainA.pdb',where="chainID='A'")

	# close the database
	db.close()