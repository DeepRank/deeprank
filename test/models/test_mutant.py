from nose.tools import eq_, ok_

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


def test_hash():
    mutant1 = PdbMutantSelection("1AK4/decoys/1AK4_cm-it0_745.pdb", "A", 10, "Q", {"A": "test/1AK4/pssm/1AK4.PSSM"})
    mutant2 = PdbMutantSelection("110M.pdb", "A", 25, "M", {"A": "110M.pssm"})

    dictionary = {mutant1: 1, mutant2: 2}

    eq_(dictionary[mutant1], 1)
    eq_(dictionary[mutant2], 2)

    ok_(hash(mutant1) != hash(mutant2))
