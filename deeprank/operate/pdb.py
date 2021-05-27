import numpy

from deeprank.models.pair import Pair
from deeprank.models.atom import Atom
from deeprank.models.residue import Residue
from deeprank.config import logger


def get_distance(position1, position2):
    """ Get euclidean distance between two positions in space.

        Args:
            position1 (numpy vector): first position
            position2 (numpy vector): second position

        Returns (float): the distance
    """

    return numpy.sqrt(numpy.sum(numpy.square(position1 - position2)))


def get_atoms(pdb2sql):
    """ Builds a list of atom objects, according to the contents of the pdb file.

        Args:
            pdb2sql (pdb2sql object): the pdb structure that we're investigating

        Returns ([Atom]): all the atoms in the pdb file.
    """

    # This is a working dictionary of residues, identified by their chains and numbers.
    residues = {}

    # This is the list of atom objects, that will be returned.
    atoms = []

    # Iterate over the atom output from pdb2sql
    for x, y, z, atom_number, atom_name, element, chain_id, residue_number, residue_name in \
            pdb2sql.get("x,y,z,rowID,name,element,chainID,resSeq,resName"):

        # Make sure that the residue is in the working directory:
        residue_id = (chain_id, residue_number)
        if residue_id not in residues:
            residues[residue_id] = Residue(residue_number, residue_name, chain_id)

        # Turn the x,y,z into a vector:
        atom_position = numpy.array([x, y, z])

        # Create the atom object and link it to the residue:
        atom = Atom(atom_number, atom_position, chain_id, atom_name, element, residues[residue_id])
        residues[residue_id].atoms.append(atom)
        atoms.append(atom)

    return atoms


def get_residue_contact_atom_pairs(pdb2sql, chain_id, residue_number, max_interatomic_distance):
    """ Find interatomic contacts around a residue.

        Args:
            pdb2sql (pdb2sql object): the pdb structure that we're investigating
            chain_id (str): the chain identifier, where the residue is located
            residue_number (int): the residue number of interest within the chain
            max_interatomic_distance (float): maximum distance between two atoms

        Returns ([Pair(int, int)]): pairs of atom numbers that contact each other
    """

    # get all the atoms in the pdb file:
    atoms = get_atoms(pdb2sql)

    # List all the atoms in the selected residue, take the coordinates while we're at it:
    residue_atoms = [atom for atom in atoms if atom.chain_id == chain_id and
                                               atom.residue.number == residue_number]
    if len(residue_atoms) == 0:
        raise ValueError("No atoms found in chain {} with residue number {}".format(chain_id, residue_number))

    # Iterate over all the atoms in the pdb, to find neighbours.
    contact_atom_pairs = set([])
    for atom in atoms:

        # Check that the atom is not one of the residue's own atoms:
        if atom.chain_id == chain_id and atom.residue.number == residue_number:
            continue

        # Within the atom iteration, iterate over the atoms in the residue:
        for residue_atom in residue_atoms:

            # Check that the atom is close enough:
            if get_distance(atom.position, residue_atom.position) < max_interatomic_distance:

                contact_atom_pairs.add(Pair(residue_atom, atom))


    return contact_atom_pairs
