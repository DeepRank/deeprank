import numpy
import logging

from deeprank.models.pair import Pair


_log = logging.getLogger(__name__)

def get_squared_distance(position1, position2):
    """Get squared euclidean distance, this is computationally cheaper that the euclidean distance

        Args:
            position1 (numpy vector): first position
            position2 (numpy vector): second position

        Returns (float): the squared distance
    """

    return numpy.sum(numpy.square(position1 - position2))


def get_distance(position1, position2):
    """ Get euclidean distance between two positions in space.

        Args:
            position1 (numpy vector): first position
            position2 (numpy vector): second position

        Returns (float): the distance
    """

    return numpy.sqrt(get_squared_distance(position1, position2))


def get_residue_contact_atom_pairs(pdb2sql, chain_id, residue_number, max_interatomic_distance):
    """ Find interatomic contacts around a residue.

        Args:
            pdb2sql (pdb2sql object): the pdb structure that we're investigating
            chain_id (str): the chain identifier, where the residue is located
            residue_number (int): the residue number of interest within the chain
            max_interatomic_distance (float): maximum distance between two atoms

        Returns ([Pair(int, int)]): pairs of atom numbers that contact each other
    """

    # Square the max distance, so that we can compare it to the squared euclidean distance between each atom pair.
    squared_max_interatomic_distance = numpy.square(max_interatomic_distance)

    # List all the atoms in the selected residue, take the coordinates while we're at it:
    residue_atoms = pdb2sql.get('rowID,x,y,z', chainID=chain_id, resSeq=residue_number)
    if len(residue_atoms) == 0:
        raise ValueError("No residue found in chain {} with number {}".format(chain_id, residue_number))

    # List all the atoms in the pdb file, take the coordinates while we're at it:
    atoms = pdb2sql.get('rowID,x,y,z')

    # Iterate over all the atoms in the pdb, to find neighbours.
    contact_atom_pairs = set([])
    for atom_nr, x, y, z in atoms:

        atom_position = numpy.array([x, y, z])

        # Within the atom iteration, iterate over the atoms in the residue:
        for residue_atom_nr, residue_x, residue_y, residue_z in residue_atoms:

            residue_position = numpy.array([residue_x, residue_y, residue_z])

            # Check that the two atom numbers are not the same and check their distance:
            if atom_nr != residue_atom_nr and \
                    get_squared_distance(atom_position, residue_position) < squared_max_interatomic_distance:

                contact_atom_pairs.add(Pair(residue_atom_nr, atom_nr))


    return contact_atom_pairs
