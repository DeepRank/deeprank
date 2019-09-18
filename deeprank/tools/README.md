# DeepRank Tools

Here are located all the generic tools used during one or across multiple steps of the workflow. A quick manual is included for the most important tools.

## PDB2SQL

The file pdb2sql.py contains a class named pdb2sql that allows using sqlite3 to manipulate PDB files. The use of SQL queries makes it very easy to extract information from the PDB file using only one line of code.

### Create a SQl data base

To create a sql database from a PDB file (named here complex.pdb) simply use:

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

### Extract information from the data base

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
### Input information into the data base


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

### Export PDB files

The method **pdb2sql.exportpdb()** allows to write pdb files. The method first call **pdb2sql.get()** with the same keyword arguments. Hence we can select part of the pdb the same way we do to extract information from the database.

```python

# write a pdb file of the chain A
sqldb.exportpdb('chainA',where="chainID='A")

```

### Close the database

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

---

## FeatureClass

The file FeatureClass.py contain a super class that all feature calculations should subclass. So far the super class only contains one method **FatureClass.export_data()** that is used to export the data of the feature to a file. This ensure that we keep the same syntax for all the features. The class has 3 attributes


  * self.type         : "Atomic" or "Residue"
  * self.feature_data : dictionary {feature_name : feature_dict}

    feature_name is the name of the feature e.g. 'coulomb' or 'vdwaals'

    feature_dict is a dictionary. The format of the key depends on the type of feature

    residue-based feature
    {(chainID, residue_name(3-letter), residue_number) : [values1, values2, ....]}

    atomic-based feature
    {(chainID, residue_name(3-letter), residue_number, atom_name) : [values1, values2, ....]}

  * self.export_directories : dictionary {feature_name : directory}

An example of feature file is given in **atomic_feature.py**. This file computes the electrostatic and vdw interactions between the contact atoms of the two chains. As you can see it subclasses the FeatureClass. All new feature should use roughly the same syntax. The new classes should fill in the feature_data and export_directories dictionary and use the export_data() method

```python

from deeprank.tools import FeatureClass

class newFeature(FeatureClass):

	def __init__(self, .... ):
		super.__init__(feature_type)
		....


	def compute_feature_1(self, .... ):
		....
		self.feature_data[name_feature_1] = dict_feature_data
		self.export_directories[name_feature_1] = export_path

	def export_data(self):
		bare_mol_name = self.pdbfile.split('/')[-1][:-4]
		super().export_data(bare_mol_name)

```

---

## Atomic Feature

The file atomic_feature.py contains a class named atomicFeature that allows computing the electrostatic interactions, van der Waals interactions and point charge of a complex. To work the class must be given:

  * a pdb file
  * a file containing atomic charges
  * a file containing the vdw parameters
  * evantually a patch file for the force field parameters

An example of use is provided in ./example/grid/atomicfeature.py.

```python
from deeprank.tools import atomicFeature

PDB = 'complex.pdb'
FF = './forcefield/'

# init the class isntance
atfeat = atomicFeature(PDB,
                       param_charge = FF + 'protein-allhdg5-4_new.top',
                       param_vdw    = FF + 'protein-allhdg5-4_new.param',
                       patch_file   = FF + 'patch.top')

# assign the force field parameters in the sqlite db
atfeat.assign_parameters()

# compute the electrostatic and vdw interactions
# between contact pairs
atfeat.evaluate_pair_interaction(print_interactions=True)

# compute the charges
# here we extand the contact atoms to
# entire residue containing at least 1 contact atom
atfeat.evaluate_charges(extend_contact_to_residue=True)

# export the data
atfeat.export_data()

# close the db
atfeat.sqldb.close()
```


In this example we compute the pair interactions and the atomic charges of the complex given in the example folder and using the force field parameters also located there. The pair interactions are outputed on the screen. For the charges, the contact atom list is extended to all the residues that contains at least one contact atom.
