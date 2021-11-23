from networkx import Graph as NetworkxGraph

class Graph(NetworkxGraph):
    "this is a graph just like in networkx, but with an id and associated target values"

    def __init__(self, id_, targets=None):
        """
            Args:
                id_(str): unique identifier for this graph
                targets(dict): the target values, keys are the target names, values are numbers
        """

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
