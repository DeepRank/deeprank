from deeprank.models.residue import Residue
from deeprank.models.pssm import Pssm
from deeprank.config.chemicals import AA_codes_pssm_ordered, AA_codes_1to3, AA_codes_3to1


def parse_old_pssm(file_):
    pssm = Pssm()
    for line in file_:
        data = line.split()
        chain_id, residue_number, residue_name = data[:3]
        values = data[3:]
        residue = Residue(residue_number, residue_name, chain_id)
        for code, value in zip(AA_codes_pssm_ordered, values):
            pssm.set_amino_acid_value(residue, code, float(value))
    return pssm


def parse_new_pssm(file_, chain_id):
    pssm = Pssm()
    header = file_.readline().split()
    for line in file_:
        data = line.split()
        record = {header[i]: data[i] for i in range(len(header))}
        residue_number = int(record['pdbresi'])
        residue_name = AA_codes_1to3[record['pdbresn']]
        residue = Residue(residue_number, residue_name, chain_id)
        for code in AA_codes_pssm_ordered:
            letter = AA_codes_3to1[code]
            value = float(record[letter])

            pssm.set_amino_acid_value(residue, code, value)

        pssm.set_information_content(residue, float(record['IC']))
    return pssm


def parse_pssm(file_, chain_id=None):
    first_line = file_.readline()
    file_.seek(0)

    if 'pdbresi' in first_line:
        if chain_id is None:
            raise ValueError("chain id is mandatory for new formatted pssm file")

        return parse_new_pssm(file_, chain_id)
    else:
        return parse_old_pssm(file_)
