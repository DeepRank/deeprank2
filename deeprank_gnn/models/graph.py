from typing import Dict, Optional
from uuid import uuid4

from networkx import Graph as NetworkxGraph


class Graph(NetworkxGraph):
    "this is a graph just like in networkx, but with an id and associated target values"

    def __init__(
        self, id_: Optional[str] = None, targets: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            id_(str, optional): unique identifier for this graph, random by default
            targets(dict, optional): the target values, keys are the target names, values are numbers
        """

        NetworkxGraph.__init__(self)
        if id_ is None:
            self._id = uuid4()
        else:
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
