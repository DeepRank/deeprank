# DeepRank Tools

Here are located all the generic tools used during one or across multiple steps of the workflow. A quick manual is included for the most important tools.

## PDB2SQL

The file pdb2sql.py contains a class named pdb2sql that allows using sqlite3 to manipulate PDB files. The use of SQL queries makes it very easy to extract information from the PDB file using only one line of code. To create a sql database from a PDB file (named here complex.pdb) simply use:

```python

from deeprank.tools import pdb2sql

# create the sql data base from the pdb file
sqldb = pdb2sql('complex.pdb')
```

After its creation the database contains 13 columns and one line for each atom in the pdb file. These columns are :

  * rowID   : the line number of each atom
  * serial  : the atom number in the pdb file
  * name    : the atom name
  * altLoc  : alternate location indicator
  * resName : the name of the residue the atom belongs to
  * chaiID  : the ID of the chain the atom belongs to
  * resSeq  : the residue number the atom belongs to
  * iCode   : Code for insertion of residue 
  * x       : x coordinate of the atom
  * y       : y coordinate of the atom
  * z       : z coordinate of the atom
  * occ     : occupancy 
  * temp    : temperature factor


To print these names use

```python
sqldb.get_colnames()
```

You can also print the entire data base using

```python
sqldb.prettyprint()
```

The **pdb2sql.get()** method can be used to extract information from the database. The get method require one or several column name as require arguments and accepts different keywords arguments to specify what we want in the data base. Here are a few examples:

``` python

# ask for the name of all the atoms
name = sqldb.get('name')

# ask for the residue numner of the TYR residues
# we use a general where sql query for that
resName = sqldb.get('resSeq',where="resName='TYR'")

# ask the reisue name of all the CA atoms
names = sqldb.get('resName',name='CA')

# ask for the position of chain A
# we here combine 3 column names with comma in between
xyz = sqldb.get('x,y,z',chain='A')

# ask the x coordinate of some atoms specified by index
# the index start at 0 !
index = [1,4,6,17,18,13]
x = sqldb.get('x',index=index)

# use a general query with multiple conditions
query = "WHERE resName = 'TYR' and chainID == 'A'"
xyz = sqldb.get('x,y,z',query=query)


```

You can also add information to a database after its creation. To do that you first need to add a new column to the database using the method **pdb2sql.add_column()**

```python
sqldb.add_column('CHARGE',coltype='FLOAT',default=0.0)
```

This will add a new column named CHARGE of type float and filled with zeros by default. To add values in this new column we recommend using the method **pdb2sql.update_column()**. Here is an example that fills the charge column with random numbers. You first need to create an array containing all the values of the column and then pass this array to the method.

```python
natom = sqldb.c.rowcount
charge = np.random.rand(natom)
sqldb.update_column('CHARGE',charge)
```

You can also input information in the database using the **pdb2sql.put()** method. The syntax is very similar to the get method. Only a single value can be passed to the method. Here are some examples:

```python

#put the charge of the chain A to 1.5
sqldb.put('CHARGE',1.5,where="chainID='A'")

# with some index
sqldb.put('CHARGE',3.5,index=[0,1,2,3])

```

The method **pdb2sql.exportpdb()** allows to write pdb files. The method first call **pdb2sql.get()** with the same keyword arguments. Hence we can select part of the pdb the same way we do to extract information from the database.

```python

# write a pdb file of the chain A
sqldb.exportpdb('chainA',where="chainID='A")

```

After using the database we can clean it and remove the .db file using

```python
# remove the .db file (default)
sqldb.close(rmdb=True)
```

or of you want to keep the .db file

```python
# keep the .db file
sqldb.close(rmdb=False)
```