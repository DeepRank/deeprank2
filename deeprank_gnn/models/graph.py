from enum import Enum
from typing import Callable, Union, List, Dict, Optional
from uuid import uuid4

import numpy
import h5py

from deeprank_gnn.models.structure import Atom, Residue
from deeprank_gnn.models.contact import Contact
from deeprank_gnn.tools.graph import graph_to_hdf5
from deeprank_gnn.models.grid import MapMethod, Grid, GridSettings
from deeprank_gnn.tools.grid import map_features, grid_to_hdf5


class Edge:
    def __init__(self, id_: Contact):
        self.id = id_
        self.features = {}

    def add_feature(self, feature_name: str, feature_function: Callable[[Contact], float]):
        feature_value = feature_function(self.id)

        self.features[feature_name] = feature_value

    @property
    def position1(self) -> numpy.array:
        return self.id.item1.position

    @property
    def position2(self) -> numpy.array:
        return self.id.item2.position


class NodeType(Enum):
    ATOM = 1
    RESIDUE = 2


class Node:
    def __init__(self, id_: Union[Atom, Residue]):
        if type(id_) == Atom:
            self._type = NodeType.ATOM

        elif type(id_) == Residue:
            self._type = NodeType.RESIDUE
        else:
            raise TypeError(type(id_))

        self.id = id_

        self.features = {}

    @property
    def type(self):
        return self._type

    def add_feature(self, feature_name: str, feature_function: Callable[[Union[Atom, Residue]], numpy.ndarray]):
        feature_value = feature_function(self.id)

        if len(feature_value.shape) != 1:
            shape_s = 'x'.join(feature_value.shape)
            raise ValueError(f"Expected a 1-dimensional array for feature {feature_name}, but got {shape_s}")

        self.features[feature_name] = feature_value

    @property
    def position(self) -> numpy.array:
        return self.id.position


class Graph:
    def __init__(self, id_: str):
        self.id = id_

        self._nodes = {}
        self._edges = {}

    def add_node(self, node: Node):
        self._nodes[node.id] = node

    def get_node(self, id_: Union[Atom, Residue]) -> Node:
        return self._nodes[id_]

    def add_edge(self, edge: Edge):
        self._edges[edge.id] = edge

    def get_edge(self, id_: Contact) -> Edge:
        return self._edges[edge.id]

    @property
    def nodes(self) -> List[Node]:
        return list(self._nodes.values())

    @property
    def edges(self) -> List[Node]:
        return list(self._edges.values())

    def map_to_grid(self, grid: Grid, method: MapMethod):

        for edge in self._edges.values():
            for feature_name, feature_value in edge.features.items():
                map_features(grid, edge.position1, feature_name, feature_value, method)
                map_features(grid, edge.position2, feature_name, feature_value, method)

        for node in self._nodes.values():
            for feature_name, feature_value in node.features.items():
                map_features(grid, node.position, feature_name, feature_value, method)

    def write_graph_to_hdf5(self, hdf5_path) -> str:
        with h5py.File(hdf5_path, 'a') as f5:
            graph_to_hdf5(self, f5)

        return hdf5_path

    def write_grid_to_hdf5(self, hdf5_path, settings: GridSettings, method: MapMethod) -> str:

        center = numpy.mean([node.position for node in self._nodes], axis=0)
        grid = Grid(self.id, settings, center)

        self.map_to_grid(grid, method)

        with h5py.File(hdf5_path, 'a') as f5:
            grid_to_hdf5(grid, f5)

        return hdf5_path
