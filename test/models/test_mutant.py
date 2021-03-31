from nose.tools import eq_

from deeprank.models.mutant import PdbMutantSelection


def test_instance():

    pdb_path = "1AK4/decoys/1AK4_cm-it0_745.pdb"
    chain_id = "A"
    residue_number = 10
    mutant_amino_acid = "Q"

    selection = PdbMutantSelection(pdb_path, chain_id, residue_number, mutant_amino_acid)

    eq_(selection.chain_id, chain_id)

    eq_(selection.residue_number, residue_number)

    eq_(selection.pdb_path, pdb_path)

    eq_(selection.mutant_amino_acid, mutant_amino_acid)
