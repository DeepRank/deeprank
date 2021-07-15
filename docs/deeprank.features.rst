Features
========

This module contains all the tools to compute feature values for molecular structure. Each submodule must be subclass ``deeprank.features.FeatureClass`` to inherit the export function. At the moment a few features have already been implemented. These are:
    - ``AtomicFeatures``:Coulomb, van der Waals interactions and atomic charges
    - ``BSA`` : Burried Surface area
    - ``FullPSSM`` : Complete PSSM data
    - ``PSSM_IC`` : Information content of the PSSM
    - ``ResidueDensity`` : The residue density for polar/apolar/charged pairs

As you can see in the source each python file contained a ``__compute_feature__`` function. This is the function called in ``deeprank.generate``.



Here are detailed the class in charge of feature calculations.


Atomic Feature
--------------

.. automodule:: deeprank.features.AtomicFeature
    :members:
    :undoc-members:
    :private-members:

.. autofunction:: deeprank.features.AtomicFeature.__compute_feature__

Buried Surface Area
-------------------

.. automodule:: deeprank.features.BSA
    :members:
    :undoc-members:
    :private-members:

.. autofunction:: deeprank.features.BSA.__compute_feature__

Energy of desolvation
-------------------

.. automodule:: deeprank.features.Edesolv
    :members:
    :undoc-members:
    :private-members:

.. autofunction:: deeprank.features.Edesolv.__compute_feature__

FullPSSM
--------

.. automodule:: deeprank.features.FullPSSM
    :members:
    :undoc-members:
    :private-members:

.. autofunction:: deeprank.features.FullPSSM.__compute_feature__


PSSM Information Content
------------------------

.. automodule:: deeprank.features.PSSM_IC
    :members:
    :undoc-members:
    :private-members:

.. autofunction:: deeprank.features.PSSM_IC.__compute_feature__


Contact Residue Density
-----------------------

.. automodule:: deeprank.features.ResidueDensity
    :members:
    :undoc-members:
    :private-members:

.. autofunction:: deeprank.features.ResidueDensity.__compute_feature__

Generic Feature Class
---------------------

.. automodule:: deeprank.features.FeatureClass
    :members:
    :undoc-members:
    :private-members:
