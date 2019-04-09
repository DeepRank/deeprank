#!/usr/bin/env python

# This script can be used to create/correct target values

import deeprank.generate.DataGenerator as DataGenerator
import os
import numpy as np
import glob
from time import time

path = './'

database = [ f for f in glob.glob(path + '*.hdf5') ]

print (database)
# create binary target

for hdf5_FL in database:
    print("Add binary class to %s" %hdf5_FL)
    data_set = DataGenerator(compute_targets  = ['deeprank.targets.binary_class'], hdf5=hdf5_FL)

    t0 = time()
    data_set.add_target(prog_bar=True)
    print('  '*25 + '--> Done in %f s.' %(time()-t0))
