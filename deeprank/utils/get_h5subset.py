#!/usr/bin/env python
"""
Extract first N groups of a hdf5 to a new hdf5 file.

Usage: python {0} <hdf5 input file> <hdf5 output file> <number of groups to write>
Example: python {0} ./001_1GPW.hdf5  ./001_1GPW_sub10.hdf5 10
"""
import sys

import h5py

USAGE = __doc__.format(__file__)


def check_input(args):
    if len(args) != 3:
        sys.stderr.write(USAGE)
        sys.exit(1)


def get_h5subset(fin, fout, n):
    """Extract first number of groups and write to a new hdf5 file.

    Args:
        fin (hdf5): input hdf5 file.
        fout (hdf5): output hdf5 file.
        n (int): first n groups to write.
    """
    n = int(n)
    h5 = h5py.File(fin, "r")
    h5out = h5py.File(fout, "w")
    print(f"First {n} groups in {fin}:")
    for i in list(h5)[0:n]:
        print(i)
        h5.copy(h5[i], h5out)

    print()
    print(f"Groups in {fout}:")
    print(list(h5out))
    h5.close()
    h5out.close()
    print()
    print(f"{fout} generated.")


if __name__ == "__main__":
    check_input(sys.argv[1:])
    fin, fout, n = sys.argv[1:]
    get_h5subset(fin, fout, n)
