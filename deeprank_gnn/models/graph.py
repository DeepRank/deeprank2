from networkx import Graph as NetworkxGraph

class Graph(NetworkxGraph):
    "this is a graph just like in networkx, but with an id"

    def __init__(self, id_, targets=None):
        NetworkxGraph.__init__(self)
        self._id = id_

        if targets is None:
            self._targets = {}
        else:
            self._targets = targets

    @property
    def id(self):
        return self._id

    @property
    def targets(self):
        return self._targets
