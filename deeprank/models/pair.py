

class Pair:
    "a hashable, comparable object for any set of two inputs where order doesn't matter"

    def __init__(self, item1, item2):
        self.item1 = item1
        self.item2 = item2

    def __hash__(self):
        return hash((self.item1, self.item2))

    def __eq__(self, other):
        return {self.item1, self.item2} == {other.item1, other.item2}

    def __iter__(self):
        return iter([self.item1, self.item2])
