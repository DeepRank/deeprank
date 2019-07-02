#!/usr/bin/env python
# Li Xue
#  2-May-2019 11:24

import sys
import re
import h5py
import os

h5FL = sys.argv[1]#'001_1GPW.hdf5' # sys.argv[1]
outDIR = sys.argv[2] # '.../'

filename = os.path.basename(h5FL)
new_h5FL = outDIR + filename

f_in = h5py.File(h5FL, 'r')
f_out = h5py.File(new_h5FL,'w')
modelIDs = list(f_in)
modelIDs

for x in modelIDs[0:5]:
    print(x)
    f_in.copy(f_in[x],f_out)
list(f_out)
f_in.close()
f_out.close()

print(f"{new_h5FL} generated.")

