

class Pair:
    """ A hashable, comparable object for any set of two inputs where order doesn't matter.
        Args:
            item1 (object): the pair's first object
            item2 (object): the pair's second object
    """

    def __init__(self, item1, item2):
        self.item1 = item1
        self.item2 = item2

    def __hash__(self):
        # The hash should be solely based on the two paired items, not on their order.
        # So rearrange the two items and turn them into a hashable tuple.
        return hash(tuple(sorted([str(self.item1), str(self.item2)])))

    def __eq__(self, other):
        # Compare the pairs as sets, so the order doesn't matter.
        return {self.item1, self.item2} == {other.item1, other.item2}

    def __iter__(self):
        # Iterate over the two items in the pair.
        return iter([self.item1, self.item2])


class ContactPair(Pair):
    "like a pair, but holds a distance too"

    def __init__(self, item1, item2, distance):
        Pair.__init__(self, item1, item2)
        self.distance = distance

    def __eq__(self, other):
        # distance must match too
        return Pair.__eq__(self, other) and self.distance == other.distance

    def __hash__(self):
        # The hash should be solely based on the two paired items, not on their order.
        # So rearrange the two items and turn them into a hashable tuple.
        return hash(tuple(sorted([str(self.item1), str(self.item2)]), self.distance))


