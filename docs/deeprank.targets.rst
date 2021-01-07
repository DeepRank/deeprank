Targets
==========================


.. automodule:: deeprank.targets


This module contains all the tools to compute target values for molecular structure. The implemented targets at the moment are:
    - ``binary_class``: Binary class ID
    - ``capri_class``: CAPRI class
    - ``dockQ``: DockQ
    - ``rmsd_fnat``: CAPRI metric IRMSD, LRMSD or FNAT

As you can see in the source each python file contained a ``__compute_feature__`` function. This is the function called in ``deeprank.generate``.



Here are detailed the class in charge of feature calculations.

Binary Class
----------------------------------------

.. automodule:: deeprank.targets.binary_class
    :members:
    :undoc-members:

CAPRI class
------------------------------

.. automodule:: deeprank.targets.capri_class
    :members:
    :undoc-members:


DockQ
------------------------------------

.. automodule:: deeprank.targets.DockQ
    :members:
    :undoc-members:


RMSD fNat
-----------------------------------

.. automodule:: deeprank.targets.rmsd_fnat
    :members:
    :undoc-members:

.. _ref_own_target:

Make your own Target
---------------------------------------
