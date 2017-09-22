import sqlite3
import subprocess as sp 
import os

'''
Class that allows to create a SL data base for a PDB file
This allows to easily etraxt information of the PDB using SQL query
So far a single SQL query has been implemented 

PDB2SQL.get(attname,where='....')


'''

class pdb2sql(object):

	'''
	CLASS that transsform  PDB file into a sqlite database
	'''

	def __init__(self,pdbfile,sqlfile='pdb2sql.db'):
		self._create_sql(pdbfile,sqlfile)

	'''
	Main function to create the SQL data base
	'''
	def _create_sql(self,pdbfile,sqlfile):

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
			print('HADDOCK FORMAT DETECTED CHAINID AT THE END')
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
	def getnames(self):
		cd = self.conn.execute('select * from atom')
		print('==> Column names are ')
		names = list(map(lambda x: x[0], cd.description))
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
	def get(self,atnames,where=None):

		'''
		Exectute a simple SQL query that extracts values of attributes for certain condition
		Ex  db.get('x,y,z',where="chainIN=='A'")
		returns an array containing the value of the attributes
		'''
		
		if where == None:
			data = [list(row) for row in self.c.execute('SELECT {an} FROM ATOM'.format(an=atnames))]
		else:
			data = [list(row) for row in self.c.execute('SELECT {an} FROM ATOM WHERE {c1}'.format(an=atnames,c1=where))]
		return data

	def get_indexes(self,atnames,index):
		strind = ','.join(map(str,index))
		return self.get(atnames,where="rowID in ({ind})".format(ind=strind))


	def exportpdb(self,fname,where=None):
		'''
		Export a PDB file for selection where
		'''
		data = self.get('*',where=where)
		f = open(fname,'w')
		for d in data:
			line = 'ATOM  '
			for iw, (colname,coltype) in enumerate(self.col.items()):
				length=max(0,self.delimiter[colname][1]-self.delimiter[colname][0]-1)
				strfmt = '{:>%d}' %(length)
				word = d[iw]
				line += strfmt.format(str(word))

				if iw>0:
					if self.delimiter[colname][0] - self.delimiter[prevcol][1] <=1 :
						nspace = self.delimiter[colname][0] - self.delimiter[prevcol][1] + 1
						line += ' '*nspace

				prevcol = colname


			line+='\n'
			f.write(line)
		f.close()

	# close the database 
	def close(self):
		self.conn.close() 



if __name__ == '__main__':

	# create the sql
	db = pdb2sql('1AK4_100w.pdb')

	db.prettyprint()

	# get the names of the columns
	db.getnames()

	# extract the xyz position of the atoms with name CB
	#xyz = db.get('x,y,z',where="chainID='A'")
	#print(xyz)

	xyz = db.get_indexes('x',index=[1,2,3])
	print(xyz)

	db.exportpdb('chainA.pdb',where="chainID='A'")

	# close the database
	db.close()