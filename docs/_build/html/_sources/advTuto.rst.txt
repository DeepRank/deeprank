Advanced Tutorial
=========================

This page gives a introduction of advanced possibilities with DeepRank

	- Create your own features
	- Create your own targets

Create your own features
--------------------------

To create your own feature you simply have to create a feature class that must subclass the ``FeatureClass`` contained in ``features/FeatureClass.py``. As an example we will create here a new feature that maps the carbon alpha of the contact residue. The first thing we need to do is to import pdb2sql and the FeatureClass superclass

>>> import pdb2sql
>>> from deeprank.features import FeatureClass

We then have to define the class and its initialization. Here we will simply initialize the class with the pdb information of the molecule we want to process. This is therefore given by

>>> # a new class based on the FeatureClass
>>> class CarbonAlphaFeature(FeatureClass):
>>>
>>> 	# init the class
>>> 	def __init__(self,pdbfile):
>>> 		super().__init__('Atomic')
>>>			self.pdb = pdb
>>>
>>>		# the feature extractor
>>>		def get_feature(self):
>>>
>>>			# create a sql database
>>>			db = pdb2sql(self.pdb)
>>>
>>>			# get the contact atoms
>>>			indA,indB = db.get_contact_atoms()
>>>			contact = indA + indB
>>>
>>>			# extract the atom keys and xyz of the contact CA atoms
>>>			ca_keys = db.get('chainID,resName,resSeq,name',name='CA',rowID=contact)
>>>			ca_xyz = db.get('x,y,z',name='CA',rowID=contact)
>>>
>>>			# create the dictionary of human readable and xyz-val data
>>>			hread, xyzval = {},{}
>>>			for key,xyz in zip(ca_keys,ca_xyz):
>>>
>>>				# human readable
>>>				# { (chainID,resName,resSeq,name) : [val] }
>>>				hread[tuple(key)] = [1.0]
>>>
>>>				# xyz-val
>>>				# { (0|1,x,y,z) : [val] }
>>>				chain = [{'A':0,'B':1}[key[0]]]
>>>				k = tuple( chain + xyz)
>>>				xyzval[k] = [1.0]
>>>
>>>			self.feature_data['CA'] = hread
>>>			self.feature_data_xyz['CA'] = xyzval


As you can see we must initialize the superclass. Since we use here the argument 'Atomic' the feature is an atomic based feature. If we would create a residue based feature (e.g. PSSM, RC, ... ) we would have used here the argument 'Residue'. This argument specifies the printing format of the feature in a human readable format and doesn't affect the mapping. From the super class the new class inherit two methods


>>> export_data_hdf5(self,featgrp)
>>> export_dataxyz_hdf5(self,featgrp)

that are used to store feature values in the HDF5 file . The class also inherits two variables

>>> self.feature_data = {}
>>> self.feature_data_xyz = {}

where we must store the feature in human readble format and xyz-val format.

To extract the feature value we are going to write a method in the class in charge of the feature extraction. This method is simply going to locate the carbon alpha and gives a value of 1 at the corresponding xyz positions.

In this function we exploit ``pdb2sql`` to locate the carbon alpha that make a contact. We then create two dicitionary where we store the feature value. The format of the human readable and xyz-val are given in comment. These two dictionnary are then added to the superclass variable ``feature_data`` and ``feature_data_xyz``.

We now must use this new class that we've just created in DeepRank. To do that we must create a function called:

>>> def __compute_feature__(pdb_data,featgrp,featgrp_raw)

Several example can be found in the feature already included in DeepRank. The location of the this function doesn't matter as we will provide the python file as value of the ``compute_features`` argument in the ``DataGenerator``. For the feature we have just created we can define that function as

>>> def __compute_feature__(pdb_data,featgrp,featgrp_raw):
>>>
>>>		cafeat = CarbonAlphaFeature(pdb_data)
>>>		cafeat.get_features()
>>>
>>>		# export in the hdf5 file
>>>		cafeat.export_dataxyz_hdf5(featgrp)
>>>		cafeat.export_data_hdf5(featgrp_raw)
>>>
>>>		# close
>>>		cafeat.db.close()


Finally to compute this feature we must call it during the data generation process. Let's assume that the file containing the ``__compute_feature__`` function is in the local folder and is called ``CAfeature.py``. To use this new feature in the generation we can simply pass the name of this file in the DataGenerator as

>>> database = DataGenerator(pdb_source=pdb_source,pdb_native=pdb_native,
>>> 	                     compute_features = ['CAFeature',....]


Create your own targets
--------------------------

The creation of new target is similar to those of new features but simpler. The targets don't need to be mapped on a grid and therefore don't need any fancy formatting. We simply need to create a new dataset in the target group of the molecule concerned. For example let's say we want to associate a random number to each conformation. To do that we can use the following code:

>>> import numpy as np
>>> 
>>> def get_random_number():
>>> 	return np.random.rand()
>>> 
>>> def __compute_target__(pdb_data,targrp):
>>> 
>>> 	target = get_random_number()
>>> 	targrp.create_dataset('FOO',data=np.array(target))

As for the features, the new target must be called in a function with a very precise name convention:

>>> def __compute_target__(pdb_data,targrp)

If as before we assume that the file containing this function is in the local folder and is called ``random.py`` we can compute the target by calling the ``DataGenerator`` with:

>>> database = DataGenerator(pdb_source=pdb_source,pdb_native=pdb_native,
>>> 	                     compute_targets = ['random',....])
