

class _PssmRecord:
    def __init__(self):
        self.amino_acid_values = {}  # uses one-letter codes as keys
        self.information_content = None


class Pssm:
    "This object stores pssm data"

    def __init__(self):
        self._residue_records = {}  # the keys should be residue identifiers

    def set_amino_acid_value(self, residue_id, amino_acid, value):
        """ Set data to the pssm object for one specific amino acid on this specific residue position

            Args:
              residue_id (Residue, unique): identifier of the residue in the protein
              amino_acid (str): three letter code of the amino acid
              value (float): specific for this amino acid on this position
        """

        if residue_id not in self._residue_records:
            self._residue_records[residue_id] = _PssmRecord()

        self._residue_records[residue_id].amino_acid_values[amino_acid] = value

    def set_information_content(self, residue_id, value):
        """ Set information content to the pssm object for one specific residue_position

            Args:
                residue_id (Residue, unique): identifier of the residue in the protein
                value (float): information content, specific for this position
        """

        if residue_id not in self._residue_records:
            self._residue_records[residue_id] = _PssmRecord()

        self._residue_records[residue_id].information_content = value

    def merge_with(self, other):
        new = Pssm()
        new._residue_records = self._residue_records
        new._residue_records.update(other._residue_records)

        return new

    def get_probability(self, residue_id, amino_acid_letter):
        """ Get the pssm's probability value of the given amino acid at the given residue position

            Args:
                residue_id (Residue, unique): identifier of the residue in the protein
                amino_acid_letter (str): one-letter code of the amino acid
        """

        if residue_id not in self._residue_records:
            raise ValueError("No such residue: {}".format(residue_id))

        return self._residue_records[residue_id].amino_acid_values[amino_acid_letter]

    def get_information_content(self, residue_id):
        """ Get the pssm's information content for a specific residue position

            Args:
                residue_id (Residue, unique): identifier of the residue in the protein
        """

        if residue_id not in self._residue_records:
            raise ValueError("No such residue: {}".format(residue_id))

        return self._residue_records[residue_id].information_content

    def items(self):
        return self._residue_records.items()

    def __len__(self):
        return len(self._residue_records)
