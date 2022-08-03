from typing import List

import numpy
from scipy.spatial import distance_matrix

from deeprankcore.models.graph import Graph, Node, Edge
from deeprankcore.models.structure import Atom, Residue
from deeprankcore.models.contact import AtomicContact, ResidueContact
from deeprankcore.domain.feature import FEATURENAME_POSITION


def build_atomic_graph( # pylint: disable=too-many-locals
    atoms: List[Atom], graph_id: str, edge_distance_cutoff: float
) -> Graph:
    """Builds a graph, using the atoms as nodes.
    The edge distance cutoff is in Ångströms.
    """

    positions = numpy.empty((len(atoms), 3))
    for atom_index, atom in enumerate(atoms):
        positions[atom_index] = atom.position

    distances = distance_matrix(positions, positions, p=2)
    neighbours = distances < edge_distance_cutoff

    graph = Graph(graph_id)
    for atom1_index, atom2_index in numpy.transpose(numpy.nonzero(neighbours)):
        if atom1_index != atom2_index:

            atom1 = atoms[atom1_index]
            atom2 = atoms[atom2_index]
            contact = AtomicContact(atom1, atom2)

            node1 = Node(atom1)
            node2 = Node(atom2)
            node1.features[FEATURENAME_POSITION] = atom1.position
            node2.features[FEATURENAME_POSITION] = atom2.position

            graph.add_node(node1)
            graph.add_node(node2)
            graph.add_edge(Edge(contact))

    return graph


def build_residue_graph( # pylint: disable=too-many-locals
    residues: List[Residue], graph_id: str, edge_distance_cutoff: float
) -> Graph:
    """Builds a graph, using the residues as nodes.
    The edge distance cutoff is in Ångströms.
    It's the shortest interatomic distance between two residues.
    """

    # collect the set of atoms
    atoms = set([])
    for residue in residues:
        for atom in residue.atoms:
            atoms.add(atom)
    atoms = list(atoms)

    positions = numpy.empty((len(atoms), 3))
    for atom_index, atom in enumerate(atoms):
        positions[atom_index] = atom.position

    distances = distance_matrix(positions, positions, p=2)
    neighbours = distances < edge_distance_cutoff

    graph = Graph(graph_id)
    for atom1_index, atom2_index in numpy.transpose(numpy.nonzero(neighbours)):
        if atom1_index != atom2_index:

            atom1 = atoms[atom1_index]
            atom2 = atoms[atom2_index]

            residue1 = atom1.residue
            residue2 = atom2.residue

            if residue1 != residue2:

                contact = ResidueContact(residue1, residue2)

                node1 = Node(residue1)
                node2 = Node(residue2)

                node1.features[FEATURENAME_POSITION] = numpy.mean(
                    [atom.position for atom in residue1.atoms], axis=0
                )
                node2.features[FEATURENAME_POSITION] = numpy.mean(
                    [atom.position for atom in residue2.atoms], axis=0
                )

                # The same residue will be added  multiple times as a node,
                # but the Graph class fixes this.
                graph.add_node(node1)
                graph.add_node(node2)
                graph.add_edge(Edge(contact))

    return graph
