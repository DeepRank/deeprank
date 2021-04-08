

class Atom:
    def __init__(self, id_, position, chain_id, name, element, residue=None):
        self.id = id_
        self.name = name
        self.element = element
        self.chain_id = chain_id
        self.position = position
        self.residue = residue

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

    def __repr__(self):
        return "Atom {} ({}) from {} {} at {}".format(self.id, self.chain_id, self.residue, self.name, self.position)

