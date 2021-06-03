from pdb2sql import pdb2sql

from deeprank.features.FeatureClass import FeatureClass
from deeprank.config.chemicals import AA_codes, AA_codes_3to1, AA_codes_1to3
from deeprank.operate.pdb import get_residue_contact_atom_pairs
from deeprank.parse.pssm import parse_pssm
from deeprank.models.pssm import Pssm
from deeprank.models.residue import Residue


IC_FEATURE_NAME = "residue_information_content"
WT_FEATURE_NAME = "wild_type_probability"
MUT_FEATURE_NAME = "mutant_probability"

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


def get_c_alpha_pos(mutant):
    pdb = pdb2sql(mutant.pdb_path)
    try:
        position = pdb.get("x,y,z", chainID=mutant.chain_id, resSeq=mutant.residue_number, name="CA")[0]

        return position
    finally:
        pdb._close()


def get_wild_type_amino_acid(mutant):
    pdb = pdb2sql(mutant.pdb_path)
    try:
        amino_acid_code = pdb.get("resName", chainID=mutant.chain_id, resSeq=mutant.residue_number)[0]

        return amino_acid_code
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

    # Get mutant probability features and place them at the C-alpha xyz position:
    c_alpha_position = get_c_alpha_pos(mutant)
    wild_type_code = get_wild_type_amino_acid(mutant)
    residue_id = Residue(mutant.residue_number, wild_type_code, mutant.chain_id)
    wild_type_probability = pssm.get_probability(residue_id, wild_type_code)
    mutant_probability = pssm.get_probability(residue_id, AA_codes_1to3[mutant.mutant_amino_acid])
    xyz_key = tuple([chain_numbers[mutant.chain_id]] + c_alpha_position)

    feature_object.feature_data_xyz[WT_FEATURE_NAME] = {xyz_key: [wild_type_probability]}
    feature_object.feature_data_xyz[MUT_FEATURE_NAME] = {xyz_key: [mutant_probability]}

    # For each neighbouring C-alpha, get the residue's PSSM features:
    feature_object.feature_data_xyz[IC_FEATURE_NAME] = {}
    for atom in neighbour_c_alphas:
        xyz_key = tuple([chain_numbers[atom.chain_id]] + atom.position)

        feature_object.feature_data_xyz[IC_FEATURE_NAME][xyz_key] = [pssm.get_information_content(atom.residue)]

    # Export to HDF5 file:
    feature_object.export_dataxyz_hdf5(feature_group)
