

class Residue:
    def __init__(self, number, name, chain_id):
        self.number = number
        self.name = name
        self.chain_id = chain_id
        self.atoms = []

    def __hash__(self):
        return hash((self.chain_id, self.number))

    def __eq__(self, other):
        return self.chain_id == other.chain_id and self.number == other.number

    def __repr__(self):
        return "Residue {} {} in {}".format(self.name, self.number, self.chain_id)
