# DeepRank Tools

Here are located all the generic tools used during one or across multiple steps of the workflow. A quick manual is included for the most important tools.

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
