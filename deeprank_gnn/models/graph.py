from enum import Enum
from typing import Callable, Union

import numpy
import h5py

from deeprank_gnn.models.feature import PointFeature
from deeprank_gnn.models.structure import Atom, Residue
from deeprank_gnn.models.contact import Contact
from deeprank_gnn.tools.graph import graph_to_hdf5
from deeprank_gnn.models.grid import MapMethod, Grid, GridSettings
from deeprank_gnn.tools.grid import map_feature, grid_to_hdf5



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
        self.id = id_
        self.features = {}

    @property
    def type(self):
        if type(self.id) == Atom:
            return NodeType.ATOM

        elif type(self.id) == Residue
            return NodeType.RESIDUE
        else:
            raise TypeError(type(self.id))

    def add_feature(self, feature_name: str, feature_function: Callable[[Union[Atom, Residue]], float]):
        feature_value = feature_function(self.id)

        self.features[feature_name] = feature_value

    @property
    def position(self) -> numpy.array:
        return self.id.position


class Graph:
    def __init__(self, id_: str, hdf5_path: str):
        self.id = id_
        self._hdf5_path = hdf5_path

        self._nodes = {}
        self._edges = {}

    def add_node(self, node):
        self._nodes[node.id] = node

    def add_edge(self, egde):
        self._edges[edge.id] = edge

    def map_to_grid(self, grid: Grid, method: MapMethod):

        edge_feature_names = list(self._edges.features.keys())
        node_feature_names = list(self._nodes.features.keys())

        for feature_name in edge_feature_names:
            point_features = []
            for edge in self._edges.values():
                point_features.append(PointFeature(edge.position1, edge.features[feature_name]))
                point_features.append(PointFeature(edge.position2, edge.features[feature_name]))
            map_feature(grid, feature_name, point_features, method)

        for feature_name in node_feature_names:
            point_features = []
            for node in self._nodes.values():
                point_features.append(PointFeature(node.position, node.features[feature_name]))
            map_feature(grid, feature_name, point_features, method)

    def to_hdf5_gnn(self) -> str:
        with h5py.File(self._hdf5_path) as f5:
            graph_to_hdf5(self, f5)

        return self._hdf5_path

    def to_hdf5_cnn(self, settings: GridSettings, method: MapMethod) -> str:

        center = numpy.mean([node.position for node in self._nodes], axis=0)
        grid = Grid(settings, center)

        self.map_to_grid(grid, method)

        with h5py.File(self._hdf5_path) as f5:
            grid_to_hdf5(self, f5)

        return self._hdf5_path
