import numpy
import logging

from deeprank.models.pair import Pair


_log = logging.getLogger(__name__)

def get_squared_distance(position1, position2):
    return numpy.sum(numpy.square(position1 - position2))


def get_distance(position1, position2):
    return numpy.sqrt(get_squared_distance(position1, position2))


def get_residue_contact_atom_pairs(pdb2sql, chain_id, residue_number, max_interatomic_distance):
    squared_max_interatomic_distance = numpy.square(max_interatomic_distance)

    residue_atoms = pdb2sql.get('rowID,x,y,z', chainID=chain_id, resSeq=residue_number)
    if len(residue_atoms) == 0:
        raise ValueError("No residue found in chain {} with number {}".format(chain_id, residue_number))

    atoms = pdb2sql.get('rowID,x,y,z')

    contact_atom_pairs = set([])

    for atom_nr, x, y, z in atoms:

        atom_position = numpy.array([x, y, z])

        for residue_atom_nr, residue_x, residue_y, residue_z in residue_atoms:

            residue_position = numpy.array([residue_x, residue_y, residue_z])

            if get_squared_distance(atom_position, residue_position) < squared_max_interatomic_distance:

                contact_atom_pairs.add(Pair(residue_atom_nr, atom_nr))


    return contact_atom_pairs
