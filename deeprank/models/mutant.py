

class PdbMutantSelection:
    """Refers to a mutant in a pdb file.

    Args:
        pdb_path (str): on disk file path to the pdb file
        chain_id (str): chain within the pdb file, where the mutant is
        residue_number (int): the identifying number of the residue within the protein chain
        mutant_amino_acid (str): one letter code of the amino acid to place at this position
        pssm_paths_by_chain (dict(str, str), optional): the paths of the pssm files per chain id, associated with the pdb file
    """

    def __init__(self, pdb_path, chain_id, residue_number, mutant_amino_acid, pssm_paths_by_chain=None):
        self._pdb_path = pdb_path
        self._chain_id = chain_id
        self._residue_number = residue_number
        self._mutant_amino_acid = mutant_amino_acid
        self._pssm_paths_by_chain = pssm_paths_by_chain

    @property
    def pdb_path(self):
        return self._pdb_path

    def has_pssm(self):
        "are the pssm files included?"
        return self._pssm_paths_by_chain is not None

    def get_pssm_chains(self):
        "returns the chain ids for which pssm files are available"
        if self._pssm_paths_by_chain is not None:
            return self._pssm_paths_by_chain.keys()
        else:
            return set([])

    def get_pssm_path(self, chain_id):
        "returns the pssm path for the given chain id"
        if self._pssm_paths_by_chain is None:
            raise ValueError("pssm paths are not set in this mutant selection")

        return self._pssm_paths_by_chain[chain_id]

    @property
    def chain_id(self):
        return self._chain_id

    @property
    def residue_number(self):
        return self._residue_number

    @property
    def mutant_amino_acid(self):
        return self._mutant_amino_acid

    def get_pssm_path(self, chain_id):
        return self._pssm_paths_by_chain[chain_id]

    def __eq__(self, other):
        return self._pdb_path == other._pdb_path and \
               self._chain_id == other._chain_id and \
               self._residue_number == other._residue_number and \
               self._mutant_amino_acid == other._mutant_amino_acid and \
               self._pssm_paths_by_chain == other._pssm_paths_by_chain
