Data Generation
===============

.. automodule:: deeprank.generate

This module contains all the tools to compute the features and targets and to map the features onto a grid of points. The main class used for the data generation is ``deeprank.generate.DataGenerator``. Through this class you can specify the molecules you want to consider, the features and the targets that need to be computed and the way to map the features on the grid. The data is stored in a single HDF5 file. In this file, each conformation has its own group that contains all the information related to the conformation. This includes the pdb data, the value of the feature (in human readable format and xyz-val format), the value of the targe values, the grid points and the mapped features on the grid.


At the moment a number of features are already implemented. This include:

    - Atomic densities
    - Coulomb & vd Waals interactions
    - Atomic charges
    - PSSM data
    - Information content
    - Buried surface area
    - Contact Residue Densities

More features can be easily implemented and integrated in the data generation workflow. You can see example here. The calculation of a number of target values have also been implemented:

    - i-RMSD
    - l-RMSD
    - FNAT
    - DockQ
    - binary class

There as well new targets can be implemented and integrated to the workflow.

Normalization of the data can be time consuming as the dataset becomes large. As an attempt to alleviate this problem, the class ``deeprank.generate.NormalizeData`` has been created. This class directly compute and store the standard deviation and mean value of each feature within a given hdf5 file.

Example:

>>> from deeprank.generate import *
>>> from time import time
>>>
>>> pdb_source     = ['./1AK4/decoys/']
>>> pdb_native     = ['./1AK4/native/']
>>> pssm_source    = './1AK4/pssm_new/'
>>> h5file = '1ak4.hdf5'
>>>
>>> #init the data assembler
>>> database = DataGenerator(chain1='C',chain2='D',
>>>                          pdb_source=pdb_source,pdb_native=pdb_native,pssm_source=pssm_source,
>>>                          data_augmentation=None,
>>>                          compute_targets  = ['deeprank.targets.dockQ'],
>>>                          compute_features = ['deeprank.features.AtomicFeature',
>>>                                              'deeprank.features.FullPSSM',
>>>                                              'deeprank.features.PSSM_IC',
>>>                                              'deeprank.features.BSA'],
>>>                          hdf5=h5file)
>>>
>>> t0 = time()
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


Structure Alignement
----------------------------------------

All the complexes contained in the dataset can be aligned similarly to facilitate and improve the training of the model. This can easily be done using the `align` option of the `DataGenerator` for example to align all the complexes along the 'z' direction one can use:

>>> database = DataGenerator(chain1='C',chain2='D',
>>>                          pdb_source=pdb_source, pdb_native=pdb_native, pssm_source=pssm_source,
>>>                          align={"axis":'z'}, data_augmentation=2,
>>>                          compute_targets=[ ... ], compute_features=[ ... ], ... )


Other options are possbile, for example if you would like to have the alignement done only using a subpart of the complex, say the chains A and B you can use :

>>> database = DataGenerator(chain1='C',chain2='D',
>>>                          pdb_source=pdb_source, pdb_native=pdb_native, pssm_source=pssm_source,
>>>                          align={"axis":'z', "selection": {"chainID":["A","B"]} }, data_augmentation=2,
>>>                          compute_targets=[ ... ], compute_features=[ ... ], ... )

All the selection offered by `pdb2sql` can be used in the `align` dictionnary e.g. : "resId":[1,2,3], "resName":['VAL','LEU'], ... Only the atoms selected will be aligned in the give direction.

You can also try to align the interface between two chains in a given plane. This can be done using :

>>> database = DataGenerator(chain1='C',chain2='D',
>>>                          pdb_source=pdb_source, pdb_native=pdb_native, pssm_source=pssm_source,
>>>                          align={"plane":'xy', "selection":"interface"}, data_augmentation=2,
>>>                          compute_targets=[ ... ], compute_features=[ ... ], ... )

which by default will use the interface between the first two chains. If you have more than two chains in the complex and want to specify which chains are forming the interface to be aligned you can use :

>>> database = DataGenerator(chain1='C',chain2='D',
>>>                          pdb_source=pdb_source, pdb_native=pdb_native, pssm_source=pssm_source,
>>>                          align={"plane":'xy', "selection":"interface", "chain1":'A', "chain2":'C'}, data_augmentation=2,
>>>                          compute_targets=[ ... ], compute_features=[ ... ], ... )

DataGenerator
----------------------------------------

.. automodule:: deeprank.generate.DataGenerator
    :members:
    :undoc-members:
    :show-inheritance:
    :private-members:

NormalizeData
----------------------------------------

.. automodule:: deeprank.generate.NormalizeData
    :members:
    :undoc-members:
    :private-members:

GridTools
------------------------------------

.. automodule:: deeprank.generate.GridTools
    :members:
    :undoc-members:
    :private-members: