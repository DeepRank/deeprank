

class PdbMutantSelection:
    """Refers to a mutant in a pdb file.

    Args:
        pdb_path (str): on disk file path to the pdb file
        chain_id (str): chain within the pdb file, where the mutant is
        residue_number (int): the identifying number of the residue within the protein chain
        mutant_amino_acid (str): one letter code of the amino acid to place at this position
    """

    def __init__(self, pdb_path, chain_id, residue_number, mutant_amino_acid):
        self._pdb_path = pdb_path
        self._chain_id = chain_id
        self._residue_number = residue_number
        self._mutant_amino_acid = mutant_amino_acid

    @property
    def pdb_path(self):
        return self._pdb_path

    @property
    def chain_id(self):
        return self._chain_id

    @property
    def residue_number(self):
        return self._residue_number

    @property
    def mutant_amino_acid(self):
        return self._mutant_amino_acid
