# 20 standard amino acids
# https://www.ncbi.nlm.nih.gov/Class/MLACourse/Modules/MolBioReview/iupac_aa_abbreviations.html
_aa_standard = [
    ('ALA', 'A'),
    ('ARG', 'R'),
    ('ASN', 'N'),
    ('ASP', 'D'),
    ('CYS', 'C'),
    ('GLN', 'Q'),
    ('GLU', 'E'),
    ('GLY', 'G'),
    ('HIS', 'H'),
    ('ILE', 'I'),
    ('LEU', 'L'),
    ('LYS', 'K'),
    ('MET', 'M'),
    ('PHE', 'F'),
    ('PRO', 'P'),
    ('SER', 'S'),
    ('THR', 'T'),
    ('TRP', 'W'),
    ('TYR', 'Y'),
    ('VAL', 'V')
    ]

# nonstandard amino acids used in blast
# https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=BlastHelp
_aa_nonstadnard = [
    ('ASX', 'B'),
    ('GLX', 'Z'),
    ('SEC', 'U')
    ]

# ordered amino acids according to PSSM file
# https://www.ncbi.nlm.nih.gov/Structure/cdd/cdd_help.shtml#CD_PSSM
_aa_pssm = ('A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V')

# ('ALA',...)
AA_codes = tuple([i[0] for i in _aa_standard])

# {'ALA': 'A', ...}
AA_codes_3to1 = dict(_aa_standard + _aa_nonstadnard)

# {'A': 'ALA', ...}
AA_codes_1to3 = dict([i[::-1] for i in _aa_standard + _aa_nonstadnard])

# AA codes ordered as PSSM header
AA_codes_pssm_ordered = tuple([AA_codes_1to3[i] for i in _aa_pssm])

# AA properties
AA_properties = {
    'ALA': 'apolar',
    'GLY': 'apolar',
    'ILE': 'apolar',
    'LEU': 'apolar',
    'MET': 'apolar',
    'PHE': 'apolar',
    'PRO': 'apolar',
    'VAL': 'apolar',
    'ARG': 'charged',
    'ASP': 'charged',
    'GLU': 'charged',
    'LYS': 'charged',
    'ASN': 'polar',
    'CYS': 'polar',
    'GLN': 'polar',
    'HIS': 'polar',
    'SER': 'polar',
    'THR': 'polar',
    'TRP': 'polar',
    'TYR': 'polar'
    }