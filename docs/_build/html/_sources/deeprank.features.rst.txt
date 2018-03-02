Feature
==========================


.. automodule:: deeprank.features


This module contains all the tools to compute feature values for molecular structure. Each submodule must be subclass ``deeprank.features.FeatureClass`` to inherit the export function. At the moment a few features have already been implemented. These are:
    - ``AtomicFeatures``:Coulomb, van der Waals interactions and atomic charges
    - ``BSA`` : Burried Surface area
    - ``NaivePSSM`` : A very simple approach for PSSM data
    - ``PSSM_IC`` : Information content of the PSSM
    - ``ResidueDensity`` : The residue density for polar/apolar/charged pairs

As you can see in the source each python file contained a ``__compute_feature__`` function. This is the function called in ``deeprank.generate``.



Here are detailled the class in charge of feature calculations.

Atomic Feature
----------------------------------------

.. automodule:: deeprank.features.AtomicFeature
    :members:
    :undoc-members:

Burried Surface Area
------------------------------

.. automodule:: deeprank.features.BSA
    :members:
    :undoc-members:


NaivePSSM
------------------------------------

.. automodule:: deeprank.features.NaivePSSM
    :members:
    :undoc-members:


Information Content
-----------------------------------

.. automodule:: deeprank.features.PSSM_IC
    :members:
    :undoc-members:


Contact Residue Density
-----------------------------------------

.. automodule:: deeprank.features.ResidueDensity
    :members:
    :undoc-members:


Generic Feature Class
---------------------------------------

.. automodule:: deeprank.features.FeatureClass
    :members:
    :undoc-members:

.. _ref_own_feature:

Make your own Feature
---------------------------------------