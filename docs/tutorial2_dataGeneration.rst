
Data Generation
===============

This section describes how to generate feature and target data using PDB files
and/or other relevant raw data, which will be directly fed to the `learning`_ step.

.. _learning: tutorial3_learning.html

The generated data is stored in a single HDF5 file. The use of HDF5 format for
data storage not only allows to save disk space through compression but also reduce I/O time during the deep learning step.

The following tutorial uses the example from the test file ``test/test_generate.py``.
All the required data, e.g. PDB files and PSSM files, can be found in the ``test/1AK4/`` directory. It's assumed that the working directory is ``test/``.


Let's start:

First import the necessary modules,

>>> from mpi4py import MPI
>>> from deeprank.generate import *

The second line imports all the submodules required for data generation.

Then, we need to set out MPI communicator that will be used later

>>> comm = MPI.COMM_WORLD

Calculating features and targets
--------------------------------

The data generation requires the PDB files of docking decoys for which we want to compute features and targets. We use ``pdb_source`` to specify where these decoy files are located,

>>> pdb_source = ['./1AK4/decoys']

which contains 5 docking decoys of the 1AK4 complex. The structure information of
these 5 decoys will be copied to the output HDF5 file.

We also need to specify the PDB files of native structures, which are required to
calculate targets like RMSD, FNAT, etc,

>>> pdb_native = ['./1AK4/native']

DeepRank will automatically look for native structure for each docking decoy by
name matching. For example, the native structure for the decoy ``./1AK4/decoys/1AK4_100w.pdb`` will be ``./1AK4/native/1AK4.pdb``.

Then, if you want to compute PSSM-related features like ``PSSM_IC``, you must specify the path to the PSSM files. The PSSM file must be named as ``<PDB_ID>.<Chain_ID>.pssm`` .

>>> pssm_source = ['./1AK4/pssm_new/']

Finally we must specify the name of the output HDF5 file,

>>> h5out = '1ak4.hdf5'

We are now ready to initialize the ``DataGenerator`` class,

>>> database = DataGenerator(pdb_source=pdb_source,
                            pdb_native=pdb_native,
                            pssm_source=pssm_source,
                            chain1='C',
                            chain2='D',
                            compute_features=['deeprank.features.AtomicFeature',
                                              'deeprank.features.FullPSSM',
                                              'deeprank.features.PSSM_IC',
                                              'deeprank.features.BSA'],
                            compute_targets=['deeprank.targets.dockQ'],
                            hdf5=h5out)

The ``compute_features`` and ``compute_targets`` are used to set which features
and targets we want to calculate by providing a list of python class names. The
available predefined features and targets can be found in the `API Reference`_.

.. _API Reference: Documentation.html

The above statement initializes a class instance but does not compute anything yet.
To actually execute the calculation, we must use the ``create_database()`` method,

>>> database.create_database(prog_bar=True)

Once you punch that DeepRank will fo through all the protein complexes specified
as input and compute all the features and targets required.

Mapping features to 3D grid
---------------------------
The next step consists in mapping the features calculated above to a grid of points centered around the molecule interface. Before mapping the features, we must define the grid first,

>>> grid_info = {'number_of_points': [30, 30, 30],
                 'resolution': [1., 1., 1.],
                 'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
                }

Here we define a grid with 30 points in x/y/z direction and a distance interval of 1Å between two neighbor points. We also provide the atomic radius for the element `C`, `N`, `O` and `S` to calculate atomic densities. Atomic density is also a type of features which have to be calculated during the mapping.

Then we start mapping the features to 3D grid,

>>> database.map_features(grid_info, try_sparse=True, prog_bar=True)

By setting ``try_sparse`` to ``True``, DeepRank will try to store the 3D grids with a built-in sparse format, which can seriously save the storage space.

Finally, it should generate the HDF5 file ``1ak4.hdf5`` that contains all the raw features and targets data and the grid-mapped data which will be used for the learning step. To easily explore these data, you could try the `DeepXplorer`_ tool.

.. _DeepXplorer: https://github.com/DeepRank/DeepXplorer

Appending features/targets
--------------------------

Suppose you've finished generating a huge HDF5 database and just realize you forgot to compute some specific features or targets. Do you have to recompute everything?

No! You can append more features and targets to the existing HDF5 database in a very simple way:

>>> h5file = '1ak4.hdf5'
>>> database = DataGenerator(compute_targets=['deeprank.targets.binary_class'],
>>>                          compute_features=['deeprank.features.ResidueDensity'],
>>>                          hdf5=h5file)
>>>
>>> # add targets
>>> database.add_target()
>>>
>>> # adda feature
>>> database.add_feature()
>>>
>>> # map features
>>> database.map_features()

Voilà! Here we simply specify the name of the existing HDF5 file we generated above, and set the new features/targets to add to this database. The methods ``add_target`` and ``add_feature`` are then called to calculate the corresponding targets and features. Don't forget to map the new features afterwards. Note that you don't have to provide any grid information for the mapping, because DeepRank will automatically detect and use the grid info that exist in the HDF5 file.