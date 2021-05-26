import logging

from pdb2sql import pdb2sql
import freesasa

from deeprank.features import FeatureClass
from deeprank.operate.pdb import get_residue_contact_atom_pairs


_log = logging.getLogger(__name__)


def get_atoms_of_iterest(mutant, distance_cutoff):

    pdb = pdb2sql(mutant.pdb_path)
    try:
        atoms = set([])
        for atom1, atom2 in get_residue_contact_atom_pairs(pdb,
                                                           mutant.chain_id,
                                                           mutant.residue_number,
                                                           distance_cutoff):

            # Add all atoms, even from the mutant residue itself:
            atoms.add(atom1)
            atoms.add(atom2)

        return atoms
    finally:
        pdb._close()


FEATURE_NAME = "accessibility"


def __compute_feature__(pdb_data, featgrp, featgrp_raw, mutant):
    "computes SASA-based features"

    # Let pdb2sql tell us which atoms are around the mutant residue:
    distance_cutoff = 8.5
    atoms_keys = set([])
    chain_ids = set([])
    for atom in get_atoms_of_iterest(mutant, distance_cutoff):
        atom_key = (atom.chain_id.strip(), int(atom.residue.number), atom.name.strip())
        atoms_keys.add(atom_key)
        chain_ids.add(atom.chain_id)

        _log.debug("contact atom: {}".format(atom_key))

    # Get structure and area data from SASA:
    structure = freesasa.Structure(mutant.pdb_path)
    result = freesasa.calc(structure)

    # Give each chain id a number:
    chain_numbers = {chain_id: index for index, chain_id in enumerate(chain_ids)}

    # Prepare a deeprank feature object:
    feature_obj = FeatureClass('Atomic')

    feature_obj.feature_data_xyz[FEATURE_NAME] = {}

    # Iterate over atoms in SASA:
    for atom_index in range(structure.nAtoms()):

        # Get atom info from SASA:
        position = structure.coord(atom_index)
        chain_id = structure.chainLabel(atom_index)
        atom_key = (chain_id.strip(),
                    int(structure.residueNumber(atom_index)),
                    structure.atomName(atom_index).strip())

        _log.debug("atom {}: {}".format(atom_index, atom_key))

        # Check that the atom is one of the selected atoms:
        if atom_key in atoms_keys:

            # Store the accessibility as a feature:
            area = result.atomArea(atom_index)

            _log.debug("  is contact atom with area = {} square Angstrom".format(area))

            xyz_key = tuple([chain_numbers[chain_id]] + position)
            feature_obj.feature_data_xyz[FEATURE_NAME][xyz_key] = [area]

    # Store the features in the hdf5 file:
    feature_obj.export_dataxyz_hdf5(featgrp)
