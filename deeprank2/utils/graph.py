from __future__ import annotations

import logging
import os
from typing import Callable

import h5py
import numpy as np
import pdb2sql.transform
from numpy.typing import NDArray
from scipy.spatial import distance_matrix

from deeprank2.domain import edgestorage as Efeat
from deeprank2.domain import nodestorage as Nfeat
from deeprank2.domain import targetstorage as targets
from deeprank2.molstruct.atom import Atom
from deeprank2.molstruct.pair import AtomicContact, Contact, ResidueContact
from deeprank2.molstruct.residue import Residue
from deeprank2.utils.grid import Augmentation, Grid, GridSettings, MapMethod

_log = logging.getLogger(__name__)


class Edge:
    def __init__(self, id_: Contact):
        self.id = id_
        self.features = {}

    def add_feature(
        self, feature_name: str, feature_function: Callable[[Contact], float]
    ):
        feature_value = feature_function(self.id)

        self.features[feature_name] = feature_value

    @property
    def position1(self) -> np.array:
        return self.id.item1.position

    @property
    def position2(self) -> np.array:
        return self.id.item2.position

    def has_nan(self) -> bool:
        """Whether there are any NaN values in the edge's features."""

        for feature_data in self.features.values():
            if np.any(np.isnan(feature_data)):
                return True

        return False


class Node:
    def __init__(self, id_: Atom | Residue):
        if isinstance(id_, Atom):
            self._type = "atom"
        elif isinstance(id_, Residue):
            self._type = "residue"
        else:
            raise TypeError(type(id_))

        self.id = id_
        self.features = {}

    @property
    def type(self):
        return self._type

    def has_nan(self) -> bool:
        """Whether there are any NaN values in the node's features."""

        for feature_data in self.features.values():
            if np.any(np.isnan(feature_data)):
                return True
        return False

    def add_feature(
        self,
        feature_name: str,
        feature_function: Callable[[Atom | Residue], NDArray],
    ):
        feature_value = feature_function(self.id)

        if len(feature_value.shape) != 1:
            shape_s = "x".join(feature_value.shape)
            raise ValueError(
                f"Expected a 1-dimensional array for feature {feature_name}, but got {shape_s}"
            )

        self.features[feature_name] = feature_value

    @property
    def position(self) -> np.array:
        return self.id.position


class Graph:
    def __init__(self, id_: str):
        self.id = id_

        self._nodes = {}
        self._edges = {}

        # targets are optional and may be set later
        self.targets = {}

        # the center only needs to be set when this graph should be mapped to a grid.
        self.center = np.array((0.0, 0.0, 0.0))

    def add_node(self, node: Node):
        self._nodes[node.id] = node

    def get_node(self, id_: Atom | Residue) -> Node:
        return self._nodes[id_]

    def add_edge(self, edge: Edge):
        self._edges[edge.id] = edge

    def get_edge(self, id_: Contact) -> Edge:
        return self._edges[id_]

    @property
    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    @property
    def edges(self) -> list[Node]:
        return list(self._edges.values())

    def has_nan(self) -> bool:
        """Whether there are any NaN values in the graph's features."""

        for node in self._nodes.values():
            if node.has_nan():
                return True

        for edge in self._edges.values():
            if edge.has_nan():
                return True

        return False

    def _map_point_features(self, grid: Grid, method: MapMethod,  # pylint: disable=too-many-arguments
                            feature_name: str, points: list[NDArray],
                            values: list[float | NDArray],
                            augmentation: Augmentation | None = None):

        points = np.stack(points, axis=0)

        if augmentation is not None:
            points = pdb2sql.transform.rot_xyz_around_axis(points,
                                                           augmentation.axis,
                                                           augmentation.angle,
                                                           self.center)

        for point_index in range(points.shape[0]):
            position = points[point_index]
            value = values[point_index]

            grid.map_feature(position, feature_name, value, method)

    def map_to_grid(self, grid: Grid, method: MapMethod, augmentation: Augmentation | None = None):

        # order edge features by xyz point
        points = []
        feature_values = {}
        for edge in self._edges.values():

            points += [edge.position1, edge.position2]

            for feature_name, feature_value in edge.features.items():
                feature_values[feature_name] = feature_values.get(feature_name, []) + [feature_value, feature_value]

        # map edge features to grid
        for feature_name, values in feature_values.items():
            self._map_point_features(grid, method, feature_name, points, values, augmentation)

        # order node features by xyz point
        points = []
        feature_values = {}
        for node in self._nodes.values():

            points.append(node.position)

            for feature_name, feature_value in node.features.items():
                feature_values[feature_name] = feature_values.get(feature_name, []) + [feature_value]

        # map node features to grid
        for feature_name, values in feature_values.items():
            self._map_point_features(grid, method, feature_name, points, values, augmentation)

    def write_to_hdf5(self, hdf5_path: str): # pylint: disable=too-many-locals
        """Write a featured graph to an hdf5 file, according to deeprank standards."""

        with h5py.File(hdf5_path, "a") as hdf5_file:

            # create groups to hold data
            graph_group = hdf5_file.require_group(self.id)
            node_features_group = graph_group.create_group(Nfeat.NODE)
            edge_feature_group = graph_group.create_group(Efeat.EDGE)

            # store node names and chain_ids
            node_names = np.array([str(key) for key in self._nodes]).astype("S")
            node_features_group.create_dataset(Nfeat.NAME, data=node_names)
            chain_ids = np.array([str(key).split()[1] for key in self._nodes]).astype("S")
            node_features_group.create_dataset(Nfeat.CHAINID, data=chain_ids)

            # store node features
            node_key_list = list(self._nodes.keys())
            first_node_data = list(self._nodes.values())[0].features
            node_feature_names = list(first_node_data.keys())
            for node_feature_name in node_feature_names:

                node_feature_data = [
                    node.features[node_feature_name] for node in self._nodes.values()
                ]

                node_features_group.create_dataset(
                    node_feature_name, data=node_feature_data
                )

            # identify edges
            edge_indices = []
            edge_names = []

            first_edge_data = list(self._edges.values())[0].features
            edge_feature_names = list(first_edge_data.keys())

            edge_feature_data = {name: [] for name in edge_feature_names}

            for edge_id, edge in self._edges.items():

                id1, id2 = edge_id
                node_index1 = node_key_list.index(id1)
                node_index2 = node_key_list.index(id2)

                edge_indices.append((node_index1, node_index2))
                edge_names.append(f"{id1}-{id2}")

                for edge_feature_name in edge_feature_names:
                    edge_feature_data[edge_feature_name].append(
                        edge.features[edge_feature_name]
                    )

            # store edge names and indices
            edge_feature_group.create_dataset(
                Efeat.NAME, data=np.array(edge_names).astype("S")
            )
            edge_feature_group.create_dataset(Efeat.INDEX, data=edge_indices)

            # store edge features
            for edge_feature_name in edge_feature_names:
                edge_feature_group.create_dataset(
                    edge_feature_name, data=edge_feature_data[edge_feature_name]
                )

            # store target values
            score_group = graph_group.create_group(targets.VALUES)
            for target_name, target_data in self.targets.items():
                score_group.create_dataset(target_name, data=target_data)

    @staticmethod
    def _find_unused_augmentation_name(unaugmented_id: str, hdf5_path: str) -> str:

        prefix = f"{unaugmented_id}_"

        entry_names_taken = []
        if os.path.isfile(hdf5_path):
            with h5py.File(hdf5_path, 'r') as hdf5_file:
                for entry_name in hdf5_file:
                    if entry_name.startswith(prefix):
                        entry_names_taken.append(entry_name)

        augmentation_count = 0
        chosen_name = f"{prefix}{augmentation_count:03}"
        while chosen_name in entry_names_taken:
            augmentation_count += 1
            chosen_name = f"{prefix}{augmentation_count:03}"

        return chosen_name

    def write_as_grid_to_hdf5(
        self, hdf5_path: str,
        settings: GridSettings,
        method: MapMethod,
        augmentation: Augmentation | None = None
    ) -> str:

        id_ = self.id
        if augmentation is not None:
            id_ = self._find_unused_augmentation_name(id_, hdf5_path)

        grid = Grid(id_, self.center.tolist(), settings)

        self.map_to_grid(grid, method, augmentation)
        grid.to_hdf5(hdf5_path)

        # store target values
        with h5py.File(hdf5_path, 'a') as hdf5_file:

            entry_group = hdf5_file[id_]

            targets_group = entry_group.require_group(targets.VALUES)
            for target_name, target_data in self.targets.items():
                if target_name not in targets_group:
                    targets_group.create_dataset(target_name, data=target_data)
                else:
                    targets_group[target_name][()] = target_data

        return hdf5_path

    def get_all_chains(self) -> list[str]:
        if isinstance(self.nodes[0].id, Residue):
            chains = set(str(res.chain).split()[1] for res in [node.id for node in self.nodes])
        elif isinstance(self.nodes[0].id, Atom):
            chains = set(str(res.chain).split()[1] for res in [node.id.residue for node in self.nodes])
        else:
            return None
        return list(chains)


    @staticmethod
    def build_graph(  # pylint: disable=too-many-locals
        nodes: list[Atom] | list[Residue],
        graph_id: str,
        max_edge_length: float,
    ) -> Graph:
        """Builds a graph.

        Args:
            nodes (list[Atom] | list[Residue]): List of `Atom`s or `Residue`s to include in graph.
                All nodes must be of same type.
            graph_id (str): Human readable identifier for graph.
            max_edge_length (float): Maximum distance between two nodes to connect them with an edge.

        Returns:
            Graph: Containing nodes (with positions) and edges.

        Raises:
            TypeError: if `nodes` argument contains a mix of different types.
        """

        if all(isinstance(node, Atom) for node in nodes):
            atoms = nodes
            NodeContact = AtomicContact
        elif all(isinstance(node, Residue) for node in nodes):
            # collect the set of atoms and remember which are on the same residue (by index)
            atoms = []
            atoms_residues = []
            for residue_index, residue in enumerate(nodes):
                for atom in residue.atoms:
                    atoms.append(atom)
                    atoms_residues.append(residue_index)
            atoms_residues = np.array(atoms_residues)
            NodeContact = ResidueContact
        else:
            raise TypeError("All nodes in the graph must be of the same type.")

        positions = np.empty((len(atoms), 3))
        for atom_index, atom in enumerate(atoms):
            positions[atom_index] = atom.position
        neighbours = max_edge_length > distance_matrix(positions, positions, p=2)

        index_pairs = np.transpose(np.nonzero(neighbours))  # atom pairs
        if NodeContact == ResidueContact:
            index_pairs = np.unique(atoms_residues[index_pairs], axis=0)  # residue pairs

        graph = Graph(graph_id)

        for index1, index2 in index_pairs:
            if index1 != index2:

                node1 = Node(nodes[index1])
                node2 = Node(nodes[index2])
                contact = NodeContact(node1.id, node2.id)

                node1.features[Nfeat.POSITION] = node1.id.position
                node2.features[Nfeat.POSITION] = node2.id.position

                # The same node will be added multiple times, but the Graph class fixes this.
                graph.add_node(node1)
                graph.add_node(node2)
                graph.add_edge(Edge(contact))

        return graph
