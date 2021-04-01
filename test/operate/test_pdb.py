import numpy
import logging
import pkg_resources
import os

from pdb2sql import pdb2sql
from nose.tools import ok_

from deeprank.operate.pdb import get_residue_contact_atom_pairs


_log = logging.getLogger(__name__)

def test_residue_contact_atoms():

    pdb_path = os.path.join(pkg_resources.resource_filename(__name__, ''),
                            "../1AK4/native/1AK4.pdb")

    try:
        pdb = pdb2sql(pdb_path)

        contact_atom_pairs = get_residue_contact_atom_pairs(pdb, 'C', 145, 8.5)

        # List all the atoms in the pairs that we found:
        contact_atoms = set([])
        for atom1, atom2 in contact_atom_pairs:
            contact_atoms.add(atom1)
            contact_atoms.add(atom2)

        # Ask pdb2sql for the residue identifiers & atom names:
        contact_atom_names = [tuple(x) for x in pdb.get("chainID,resSeq,name", rowID=list(contact_atoms))]
    finally:
        pdb._close()

    # Now, we need to verify that the function "get_residue_contact_atom_pairs" returned the right pairs.
    # We do so by selecting one close residue and one distant residue.

    neighbour = ('C', 144, 'CA')  # this residue sits right next to the selected residue
    distant = ('B', 134, 'OE2')  # this residue is very far away

    # Check that the close residue is present in the list and that the distant residue is absent.

    ok_(neighbour in contact_atom_names)

    ok_(distant not in contact_atom_names)
