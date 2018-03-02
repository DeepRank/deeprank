Generate Data
==========================

.. automodule:: deeprank.generate

This module contains all the tools to compute the features and targets and to map the features onto a grid of points. The main class used for the data generation is ``deeprank.generate.DataGenerator``. Through this class you can specify the molecules you want to consider, the features and the targets that need to be computed and the way to map the features on the grid. The data is stored in a single HDF5 file. In this file, each conformation has its own group that contains all the information related to the conformation. This includes, the pdb data, the value of the feature (in human readable format and xyz-val format), the value of the targe values, the grid points and the mapped features on the grid.


At the moment a number of features are already implemented. This inculde:

    - Atomic densities
    - Coulomb & vd Waals interactions
    - Atomic charges
    - PSSM data
    - Information content
    - Burried surface area
    - Contact Residue Densities

More features can be easily implemented and integrated in the data generation workflow. You can see example here: :ref:`_ref_own_feature`. The calculation of a number of target values have also been implemented:

    - i-RMSD
    - l-RMSD
    - FNAT
    - DockQ
    - binary class
    - Haddock score

There as well new targets can be implemented and integrated to the workflow.

Normalization of the data can be time consuming as the dataset becomes large. As an attempt to alleviate this problem, the class ``deeprank.generate.NormalizeData`` has been created. This class directly compute and store the standard deviation and mean value of each feature within a given hdf5 file.

Example:

>>> from deeprank.generate import *
>>> pdb_source     = ['./1AK4/decoys/']
>>> pdb_native     = ['./1AK4/native/']
>>> h5file = '1ak4.hdf5'
>>>
>>> #init the data assembler
>>> database = DataGenerator(pdb_source=pdb_source,pdb_native=pdb_native,data_augmentation=None,
>>>                          compute_targets  = ['deeprank.targets.dockQ'],
>>>                          compute_features = ['deeprank.features.AtomicFeature',
>>>                                              'deeprank.features.NaivePSSM',
>>>                                              'deeprank.features.PSSM_IC',
>>>                                              'deeprank.features.BSA'],
>>>                          hdf5=h5file)
>>>
>>> #create new files
>>> database.create_database(prog_bar=True)
>>>
>>> # map the features
>>> grid_info = {
>>>     'number_of_points' : [30,30,30],
>>>     'resolution' : [1.,1.,1.],
>>>     'atomic_densities' : {'CA':3.5,'N':3.5,'O':3.5,'C':3.5},
>>> }
>>> database.map_features(grid_info,try_sparse=True,time=False,prog_bar=True)
>>>
>>> # add a new target
>>> database.add_target(prog_bar=True)
>>> print(' '*25 + '--> Done in %f s.' %(time()-t0))
>>>
>>> # get the normalization
>>> norm = NormalizeData(h5file)
>>> norm.get()

The details of the different submodule are listed here. The only module that really needs to be used is ``DataGenerator`` and ``NormalizeData``. The ``GridTools`` class should not be directly used by inexperienced users.

DataGenerator
----------------------------------------

.. automodule:: deeprank.generate.DataGenerator
    :members:
    :undoc-members:
    :show-inheritance:

NormalizeData
----------------------------------------

.. automodule:: deeprank.generate.NormalizeData
    :members:
    :undoc-members:

GridTools
------------------------------------

.. automodule:: deeprank.generate.GridTools
    :members:
    :undoc-members:

