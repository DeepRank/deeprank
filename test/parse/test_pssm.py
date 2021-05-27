from nose.tools import eq_

from deeprank.models.pssm import Pssm
from deeprank.models.residue import Residue
from deeprank.parse.pssm import parse_pssm


def test_parse_old():
    with open("test/1AK4/pssm/1AK4.PSSM", 'rt') as f:
        pssm = parse_pssm(f)

    eq_(len(pssm), 310)
    for residue, record in pssm.items():
        eq_(type(residue), Residue)
        for aa, value in record.amino_acid_values.items():
            eq_(type(aa), str)
            eq_(type(value), float)
        eq_(record.information_content, None)


def test_parse_new():
    pssm = Pssm()
    for chain_id in ['C', 'D']:
        with open("test/1AK4/pssm_new/1AK4.%s.pssm" % chain_id, 'rt') as f:
            pssm.merge_with(parse_pssm(f, chain_id))

    eq_(len(pssm), 310)
    for residue, record in pssm.items():
        eq_(type(residue), Residue)
        for aa, value in record.amino_acid_values.items():
            eq_(type(aa), str)
            eq_(type(value), float)
        eq_(type(record.information_content), float)
