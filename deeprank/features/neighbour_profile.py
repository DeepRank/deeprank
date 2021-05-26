from pdb2sql import pdb2sql

from deeprank.features.FeatureClass import FeatureClass
from deeprank.config.chemicals import AA_codes
from deeprank.operate.pdb import get_residue_contact_atom_pairs
from deeprank.parse.pssm import parse_pssm
from deeprank.models.pssm import Pssm


IC_FEATURE_NAME = "residue_information_content"

def get_probability_feature_name(amino_acid_code):
    return "residue_%s_probability" % amino_acid_code


def get_neighbour_c_alphas(mutant, distance_cutoff):
    pdb = pdb2sql(mutant.pdb_path)
    try:
        atoms = set([])
        for atom1, atom2 in get_residue_contact_atom_pairs(pdb, mutant.chain_id, mutant.residue_number, distance_cutoff):

            # For each residue in the contact range, get the C-alpha:
            for atom in (atom1.residue.atoms + atom2.residue.atoms):
                if atom.name == "CA":
                    atoms.add(atom1)

        return atoms
    finally:
        pdb._close()


def _get_pssm(chain_ids, mutant):
    pssm = Pssm()
    for chain_id in chain_ids:
        with open(mutant.get_pssm_path(chain_id), 'rt', encoding='utf_8') as f:
            pssm.merge_with(parse_pssm(f, chain_id))
    return pssm


def __compute_feature__(pdb_data, feature_group, raw_feature_group, mutant):
    "this feature module adds amino acid probability and residue information content as deeprank features"

    # Get the C-alpha atoms, each belongs to a neighbouring residue
    neighbour_c_alphas = get_neighbour_c_alphas(mutant, 8.5)

    # Give each chain id a number:
    chain_ids = set([atom.chain_id for atom in neighbour_c_alphas])
    chain_numbers = {chain_id: index for index, chain_id in enumerate(chain_ids)}

    pssm = _get_pssm(chain_ids, mutant)

    # Initialize a feature object:
    feature_object = FeatureClass("Residue")
    for code in AA_codes:
        feature_object.feature_data_xyz[get_probability_feature_name(code)] = {}
    feature_object.feature_data_xyz[IC_FEATURE_NAME] = {}

    # For each neighbouring C-alpha, get the residue's PSSM features:
    for atom in neighbour_c_alphas:
        xyz_key = tuple([chain_numbers[atom.chain_id]] + atom.position)

        for code in AA_codes:

            feature_name = get_probability_feature_name(code)

            feature_object.feature_data_xyz[feature_name][xyz_key] = [pssm.get_probability(atom.residue, code)]

        feature_object.feature_data_xyz[IC_FEATURE_NAME][xyz_key] = [pssm.get_information_content(atom.residue)]

    # Export to HDF5 file:
    feature_object.export_dataxyz_hdf5(feature_group)
