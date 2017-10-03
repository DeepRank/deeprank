# DeepRank Tools

Here are located all the generic tools used during one or across multiple steps of the workflow. A quick manual is included for the most important tools.

## PDB2SQL

The file pdb2sql.py contains a class named pdb2sql that allows using sqlite3 to manipulate PDB files. The use of SQL queries makes it very easy to extract information from the PDB file using only one line of code. Here is a small example

```python

from deeprank.tools import pdb2sql

# create the sql data base from the pdb file
sqldb = pdb2sql('complex.pdb')

# ask for the position of chain A
xyz = sqldb.get('x,y,z',chain='A')

```