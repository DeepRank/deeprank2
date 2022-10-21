from uuid import uuid4

from pdb2sql import pdb2sql
import numpy

from deeprankcore.models.structure import Chain, Atom
from deeprankcore.models.contact import AtomicContact, ResidueContact
from deeprankcore.models.graph import Edge, Graph
from deeprankcore.tools.pdb import get_structure
from deeprankcore.feature.atomic_contact import add_features
from deeprankcore.models.amino_acid import alanine
from deeprankcore.models.variant import SingleResidueVariant
from deeprankcore.domain.features import edgefeats as Efeat



def _get_atom(chain: Chain, residue_number: int, atom_name: str) -> Atom:

    for residue in chain.residues:
        if residue.number == residue_number:
            for atom in residue.atoms:
                if atom.name == atom_name:
                    return atom

    raise ValueError(
        f"Not found: chain {chain.id} residue {residue_number} atom {atom_name}"
    )


def _wrap_in_graph(edge: Edge):
    g = Graph(uuid4().hex)
    g.add_edge(edge)
    return g


def test_add_features():

    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101m")
    finally:
        pdb._close() # pylint: disable=protected-access

    variant = SingleResidueVariant(structure.chains[0].residues[10], alanine)

    # MET 0: N - CA, very close, should have positive vanderwaals energy
    contact = AtomicContact(
        _get_atom(structure.chains[0], 0, "N"), _get_atom(structure.chains[0], 0, "CA")
    )
    edge_close = Edge(contact)
    add_features(pdb_path, _wrap_in_graph(edge_close), variant)
    assert not numpy.isnan(edge_close.features[Efeat.VANDERWAALS])
    assert edge_close.features[Efeat.VANDERWAALS] > 0.0, edge_close.features[
        Efeat.VANDERWAALS
    ]

    # MET 0 N - ASP 27 CB, very far, should have negative vanderwaals energ
    contact = AtomicContact(
        _get_atom(structure.chains[0], 0, "N"), _get_atom(structure.chains[0], 27, "CB")
    )
    edge_far = Edge(contact)
    add_features(pdb_path, _wrap_in_graph(edge_far), variant)
    assert not numpy.isnan(edge_far.features[Efeat.VANDERWAALS])
    assert edge_far.features[Efeat.VANDERWAALS] < 0.0, edge_far.features[
        Efeat.VANDERWAALS
    ]

    # MET 0 N - PHE 138 CG, intermediate distance, should have more negative
    # vanderwaals energy than the war interaction
    contact = AtomicContact(
        _get_atom(structure.chains[0], 0, "N"),
        _get_atom(structure.chains[0], 138, "CG"),
    )
    edge_intermediate = Edge(contact)
    add_features(pdb_path, _wrap_in_graph(edge_intermediate), variant)
    assert not numpy.isnan(edge_intermediate.features[Efeat.VANDERWAALS])
    assert (
        edge_intermediate.features[Efeat.VANDERWAALS]
        < edge_far.features[Efeat.VANDERWAALS]
    ), f"{edge_intermediate.features[Efeat.VANDERWAALS]} >= {edge_far.features[Efeat.VANDERWAALS]}"

    # Check the distances
    assert (
        edge_close.features[Efeat.DISTANCE]
        < edge_intermediate.features[Efeat.DISTANCE]
    ), f"{edge_close.features[Efeat.DISTANCE]} >= {edge_intermediate.features[Efeat.DISTANCE]}"
    assert (
        edge_far.features[Efeat.DISTANCE]
        > edge_intermediate.features[Efeat.DISTANCE]
    ), f"{edge_far.features[Efeat.DISTANCE]} <= {edge_intermediate.features[Efeat.DISTANCE]}"

    # ARG 139 CZ - GLU 136 OE2, very close attractive electrostatic energy
    contact = AtomicContact(
        _get_atom(structure.chains[0], 139, "CZ"),
        _get_atom(structure.chains[0], 136, "OE2"),
    )
    close_attracting_edge = Edge(contact)
    add_features(pdb_path, _wrap_in_graph(close_attracting_edge), variant)
    assert not numpy.isnan(close_attracting_edge.features[Efeat.ELECTROSTATIC])
    assert (
        close_attracting_edge.features[Efeat.ELECTROSTATIC] < 0.0
    ), close_attracting_edge.features[Efeat.ELECTROSTATIC]

    # ARG 139 CZ - ASP 20 OD2, far attractive electrostatic energy
    contact = AtomicContact(
        _get_atom(structure.chains[0], 139, "CZ"),
        _get_atom(structure.chains[0], 20, "OD2"),
    )
    far_attracting_edge = Edge(contact)
    add_features(pdb_path, _wrap_in_graph(far_attracting_edge), variant)
    assert not numpy.isnan(far_attracting_edge.features[Efeat.ELECTROSTATIC])
    assert (
        far_attracting_edge.features[Efeat.ELECTROSTATIC] < 0.0
    ), far_attracting_edge.features[Efeat.ELECTROSTATIC]
    assert (
        far_attracting_edge.features[Efeat.ELECTROSTATIC]
        > close_attracting_edge.features[Efeat.ELECTROSTATIC]
    ), f"{far_attracting_edge.features[Efeat.ELECTROSTATIC]} <= {close_attracting_edge.features[Efeat.ELECTROSTATIC]}"

    # GLU 109 OE2 - GLU 105 OE1, repulsive electrostatic energy
    contact = AtomicContact(
        _get_atom(structure.chains[0], 109, "OE2"),
        _get_atom(structure.chains[0], 105, "OE1"),
    )
    opposing_edge = Edge(contact)
    add_features(pdb_path, _wrap_in_graph(opposing_edge), variant)
    assert not numpy.isnan(opposing_edge.features[Efeat.ELECTROSTATIC])
    assert (
        opposing_edge.features[Efeat.ELECTROSTATIC] > 0.0
    ), opposing_edge.features[Efeat.ELECTROSTATIC]

    # check that we can calculate residue contacts
    contact = ResidueContact(
        structure.chains[0].residues[0], structure.chains[0].residues[1]
    )
    edge = Edge(contact)
    add_features(pdb_path, _wrap_in_graph(edge), variant)
    assert not numpy.isnan(edge.features[Efeat.DISTANCE]) > 0.0
    assert edge.features[Efeat.DISTANCE] > 0.0
    assert edge.features[Efeat.DISTANCE] < 1e5

    assert not numpy.isnan(edge.features[Efeat.ELECTROSTATIC])
    assert edge.features[Efeat.ELECTROSTATIC] != 0.0, edge.features[
        Efeat.ELECTROSTATIC
    ]

    assert not numpy.isnan(edge.features[Efeat.VANDERWAALS])
    assert edge.features[Efeat.VANDERWAALS] != 0.0, edge.features[
        Efeat.VANDERWAALS
    ]
