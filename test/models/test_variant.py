from nose.tools import eq_, ok_

from deeprank.models.variant import PdbVariantSelection


def test_instance():

    pdb_path = "1AK4/decoys/1AK4_cm-it0_745.pdb"
    chain_id = "A"
    residue_number = 10
    amino_acid = "Q"

    selection = PdbVariantSelection(pdb_path, chain_id, residue_number, amino_acid)

    eq_(selection.chain_id, chain_id)

    eq_(selection.residue_number, residue_number)

    eq_(selection.pdb_path, pdb_path)

    eq_(selection.amino_acid, amino_acid)


def test_hash():
    variant1 = PdbVariantSelection("1AK4/decoys/1AK4_cm-it0_745.pdb", "A", 10, "Q", {"A": "test/1AK4/pssm/1AK4.PSSM"})
    variant2 = PdbVariantSelection("110M.pdb", "A", 25, "M", {"A": "110M.pssm"})

    dictionary = {variant1: 1, variant2: 2}

    eq_(dictionary[variant1], 1)
    eq_(dictionary[variant2], 2)

    ok_(hash(variant1) != hash(variant2))
