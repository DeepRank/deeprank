import numpy
import logging

from pdb2sql import pdb2sql
from nose.tools import ok_

from deeprank.operate.pdb import get_residue_contact_atom_pairs


_log = logging.getLogger(__name__)

def test_residue_contact_atoms():
    pdb_path = "test/1AK4/native/1AK4.pdb"

    try:
        pdb = pdb2sql(pdb_path)

        contact_atom_pairs = get_residue_contact_atom_pairs(pdb, 'C', 145, 8.5)

        contact_atoms = set([])
        for atom1, atom2 in contact_atom_pairs:
            contact_atoms.add(atom1)
            contact_atoms.add(atom2)

        contact_atom_names = [tuple(x) for x in pdb.get("chainID,resSeq,name", rowID=list(contact_atoms))]
    finally:
        pdb._close()

    neighbour = ('C', 144, 'CA')
    distant = ('B', 134, 'OE2')

    ok_(neighbour in contact_atom_names)

    ok_(distant not in contact_atom_names)
