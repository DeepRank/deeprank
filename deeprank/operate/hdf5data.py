from deeprank.models.mutant import PdbMutantSelection


def store_mutant(molecule_group, mutant):
    molecule_group.attrs['pdb_path'] = mutant.pdb_path

    for chain_id in mutant.get_pssm_chains():
        molecule_group.attrs['pssm_path_%s' % chain_id] = mutant.get_pssm_path(chain_id)

    molecule_group.attrs['mutant_chain_id'] = mutant.chain_id

    molecule_group.attrs['mutant_residue_number'] = mutant.residue_number

    molecule_group.attrs['mutant_amino_acid'] = mutant.mutant_amino_acid


def load_mutant(molecule_group):
    pdb_path = molecule_group.attrs['pdb_path']

    pssm_paths_by_chain = {}
    for attr_name in molecule_group.attrs:
        if attr_name.startswith("pssm_path_"):
            chain_id = attr_name.split('_')[-1]
            pssm_paths_by_chain[chain_id] = molecule_group.attrs[attr_name]

    chain_id = molecule_group.attrs['mutant_chain_id']

    residue_number = molecule_group.attrs['mutant_residue_number']

    amino_acid = molecule_group.attrs['mutant_amino_acid']

    mutant = PdbMutantSelection(pdb_path, chain_id, residue_number, amino_acid, pssm_paths_by_chain)

    return mutant
