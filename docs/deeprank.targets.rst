Target
==========================


.. automodule:: deeprank.targets


This module contains all the functions to compute target values for molecular structures. The implemented targets at the moment are:
    - ``binary_class``: Binary class ID
    - ``capri_class``: CAPRI class
    - ``dockQ``: DockQ
    - ``rmsd_fnat``: CAPRI metric IRMSD, LRMSD or FNAT

As you can see in the source each python file contained a ``__compute_target__`` function. This is the function called in ``deeprank.generate``.



Here are detailed the class in charge of target calculations.

Binary Class
----------------------------------------

.. automodule:: deeprank.targets.binary_class.__compute_target__
    :members:
    :undoc-members:

CAPRI class
----------------------------------------

.. automodule:: deeprank.targets.capri_class.__compute_target__
    :members:
    :undoc-members:


DockQ
----------------------------------------

.. automodule:: deeprank.targets.dockQ.__compute_target__
    :members:
    :undoc-members:


RMSD fNat
----------------------------------------

.. automodule:: deeprank.targets.rmsd_fnat.__compute_target__
    :members:
    :undoc-members:
