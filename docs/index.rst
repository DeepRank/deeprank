.. DeepRank documentation master file, created by
   sphinx-quickstart on Mon Feb 26 14:44:07 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DeepRank
========

`DeepRank`_ is a general, configurable deep learning framework for data mining
protein-protein interactions (PPIs) using 3D convolutional neural networks (CNNs).

DeepRank contains useful APIs for pre-processing PPIs data, computing features and
targets, as well as training and testing CNN models.

.. _`DeepRank`: https://github.com/DeepRank/deeprank

**DeepRank highlights**:

- Predefined atom-level and residue-level PPI feature types
   - *e.g. atomic density, vdw energy, residue contacts, PSSM, etc.*
- Predefined target types
   - *e.g. binary class, CAPRI categories, DockQ, RMSD, FNAT, etc.*
- Flexible definition of both new features and targets
- 3D grid feature mapping
- Efficient data storage in HDF5 format
- Support both classification and regression (based on PyTorch)

Tutorial
--------

.. toctree::
   :maxdepth: 3

   tutorial1_installing
   tutorial2_dataGeneration
   tutorial3_learning
   tutorial4_advanced

API Reference
-------------

.. toctree::
   :maxdepth: 3

   Documentation

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
