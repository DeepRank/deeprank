

class Residue:
    def __init__(self, number, name):
        self.number = number
        self.name = name
        self.atoms = []

    @property
    def chain_id(self):
        chain_ids = set([atom.chain_id for atom in self.atoms])
        if len(chain_ids) > 1:
            raise ValueError("residue {} {} contains atoms of different chains: {}".format(self.name, self.number, self.atoms))

        return list(chain_ids)[0]

    def __hash__(self):
        return hash((self.chain_id, self.number))

    def __eq__(self, other):
        return self.chain_id == other.chain_id and self.number == other.number

    def __repr__(self):
        return "Residue {} {}".format(self.name, self.number)
