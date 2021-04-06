import numpy
import logging
import pkg_resources
import os

from pdb2sql import pdb2sql
from nose.tools import ok_

from deeprank.operate.pdb import get_residue_contact_atom_pairs, get_atoms


_log = logging.getLogger(__name__)


def test_get_atoms():
    pdb_path = os.path.join(pkg_resources.resource_filename(__name__, ''),
                            "../1AK4/native/1AK4.pdb")

    try:
        pdb = pdb2sql(pdb_path)

        atoms = get_atoms(pdb)

        ok_(len(atoms) > 0)
    finally:
        pdb._close()


def _find_atom(atoms, chain_id, residue_number, name):
    matching_atoms = [atom for atom in atoms if atom.chain_id == chain_id and
                                                atom.name == name and atom.residue.number == residue_number]

    assert len(matching_atoms) == 1, "Expected exacly one matching atom, got {}".format(len(matching_atoms))

    return matching_atoms[0]


def test_residue_contact_atoms():

    pdb_path = os.path.join(pkg_resources.resource_filename(__name__, ''),
                            "../1AK4/native/1AK4.pdb")

    try:
        pdb = pdb2sql(pdb_path)

        atoms = get_atoms(pdb)

        contact_atom_pairs = get_residue_contact_atom_pairs(pdb, 'D', 145, 8.5)

        # List all the atoms in the pairs that we found:
        contact_atoms = set([])
        for atom1, atom2 in contact_atom_pairs:
            contact_atoms.add(atom1)
            contact_atoms.add(atom2)
    finally:
        pdb._close()

    # Now, we need to verify that the function "get_residue_contact_atom_pairs" returned the right pairs.
    # We do so by selecting one close residue and one distant residue.

    neighbour = _find_atom(atoms, 'D', 144, 'CA')  # this residue sits right next to the selected residue
    distant = _find_atom(atoms, 'C', 134, 'OE2')  # this residue is very far away

    # Check that the close residue is present in the list and that the distant residue is absent.

    ok_(neighbour in contact_atoms)

    ok_(distant not in contact_atoms)
