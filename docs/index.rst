.. DeepRank documentation file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DeepRank
========

`DeepRank v.2.0`_ is a configurable deep learning framework to predict 
pathogenicity of missense variants using 3D convolutional neural networks (CNNs).

DeepRank contains useful APIs for pre-processing protein structural data, computing features and
targets, as well as training and testing CNN models.

.. _`DeepRank v.2.0`: https://github.com/DeepRank/deeprank/tree/efro-project

**DeepRank highlights**:

- Predefined atom-level and residue-level structural feature types
   - *e.g. residue contacts, atomic density, vdw energy, solvent accessibility, PSSM.*
- Predefined target types
   - *e.g. Benign or Pathogenic*
- Flexible definition of both new features and targets
- 3D grid feature mapping
- Efficient data storage in HDF5 format
- Support both classification and regression (based on PyTorch)

Tutorial
--------

.. toctree::
   :maxdepth: 3

   tutorial

API Reference
-------------

.. toctree::
   :maxdepth: 3

   Documentation

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
