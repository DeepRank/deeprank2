from typing import List

import numpy
from scipy.spatial import distance_matrix

from deeprank_gnn.models.graph import Graph, Node, Edge
from deeprank_gnn.models.structure import Atom, Residue
from deeprank_gnn.models.contact import AtomicContact, ResidueContact


def build_atomic_graph(atoms: List[Atom], graph_id: str, edge_distance_cutoff: float) -> Graph:
    """ Builds a graph, using the atoms as nodes.
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

            graph.add_node(Node(atom1))
            graph.add_node(Node(atom2))
            graph.add_edge(Edge(contact))

    return graph


def build_residue_graph(residues: List[Residue], graph_id: str, edge_distance_cutoff: float) -> Graph:
    """ Builds a graph, using the residues as nodes.
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

            contact = ResidueContact(residue1, residue2)

            graph.add_node(Node(residue1))
            graph.add_node(Node(residue2))
            graph.add_edge(Edge(contact))

    return graph
