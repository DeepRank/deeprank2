from networkx import Graph as NetworkxGraph

class Graph(NetworkxGraph):
    "this is a graph just like in networkx, but with an id"

    def __init__(self, id_):
        NetworkxGraph.__init__(self)
        self._id = id_

    @property
    def id(self):
        return self._id
