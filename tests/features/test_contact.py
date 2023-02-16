from uuid import uuid4
from pdb2sql import pdb2sql
import numpy as np
from scipy.spatial import distance_matrix
from deeprankcore.molstruct.structure import Chain, PDBStructure
from deeprankcore.molstruct.atom import Atom
from deeprankcore.molstruct.pair import AtomicContact, ResidueContact
from deeprankcore.molstruct.variant import SingleResidueVariant
from deeprankcore.utils.graph import Edge, Graph
from deeprankcore.utils.buildgraph import get_structure
from deeprankcore.features.contact import add_features, get_bonded_matrix
from deeprankcore.domain.aminoacidlist import alanine
from deeprankcore.domain import edgestorage as Efeat



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


def test_within_3bonds_distinction():

    pdb_path = "tests/data/pdb/1ak4/1ak4.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101m")
    finally:
        pdb._close() # pylint: disable=protected-access

    chain_C = structure.get_chain('C')
    chain_D = structure.get_chain('D')

    atoms = structure.get_atoms()
    positions = np.array([atom.position for atom in atoms])

    count_atoms = len(atoms)

    BONDED = get_bonded_matrix(positions, 2.0, 3)

    assert BONDED.shape == (count_atoms, count_atoms)

    index_C_phe60_CE1 = atoms.index(_get_atom(chain_C, 60, "CE1"))
    index_C_trp121_CZ2 = atoms.index(_get_atom(chain_C, 121, "CZ2"))
    index_C_asn102_O = atoms.index(_get_atom(chain_C, 102, "O"))
    index_D_leu111_CG = atoms.index(_get_atom(chain_D, 111, "CG"))
    index_D_pro93_CA = atoms.index(_get_atom(chain_D, 93, "CA"))
    index_D_pro93_CB = atoms.index(_get_atom(chain_D, 93, "CB"))
    index_D_pro93_CG = atoms.index(_get_atom(chain_D, 93, "CG"))
    index_D_pro93_CD = atoms.index(_get_atom(chain_D, 93, "CD"))
    index_D_ala92_CA = atoms.index(_get_atom(chain_D, 92, "CA"))
    index_D_ala92_CB = atoms.index(_get_atom(chain_D, 92, "CB"))
    index_D_gly89_N = atoms.index(_get_atom(chain_D, 89, "N"))

    # one bond away
    assert BONDED[index_D_pro93_CA, index_D_pro93_CB]
    assert BONDED[index_D_pro93_CB, index_D_pro93_CA]

    # two bonds away
    assert BONDED[index_D_pro93_CA, index_D_pro93_CG]
    assert BONDED[index_D_pro93_CG, index_D_pro93_CA]

    # three bonds away
    assert BONDED[index_D_pro93_CA, index_D_ala92_CA]
    assert BONDED[index_D_ala92_CA, index_D_pro93_CA]

    # four bonds away
    assert not BONDED[index_D_pro93_CA, index_D_ala92_CB]

    # in different chain, but hydrogen bonded
    assert not BONDED[index_D_gly89_N, index_C_asn102_O]

    # close, but not connected
    assert not BONDED[index_C_trp121_CZ2, index_C_phe60_CE1]

    # far away from each other
    assert not BONDED[index_D_leu111_CG, index_D_pro93_CA]


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
    assert not np.isnan(edge_close.features[Efeat.VANDERWAALS])
    assert edge_close.features[Efeat.VANDERWAALS] > 0.0, edge_close.features[
        Efeat.VANDERWAALS
    ]

    # MET 0: N - VAL 1 CB, should be within the cutoff distance and have negative vanderwaals energy
    contact = AtomicContact(
        _get_atom(structure.chains[0], 0, "N"), _get_atom(structure.chains[0], 1, "CB")
    )
    edge_far = Edge(contact)
    add_features(pdb_path, _wrap_in_graph(edge_far), variant)
    assert not np.isnan(edge_far.features[Efeat.VANDERWAALS])
    assert edge_far.features[Efeat.VANDERWAALS] < 0.0, edge_far.features[
        Efeat.VANDERWAALS
    ]

    # MET 0 N - ASP 27 CB, too far, should have zero vanderwaals energy
    contact = AtomicContact(
         _get_atom(structure.chains[0], 0, "N"), _get_atom(structure.chains[0], 27, "CB")
    )
    edge_out_of_range = Edge(contact)
    add_features(pdb_path, _wrap_in_graph(edge_out_of_range), variant)
    assert not np.isnan(edge_out_of_range.features[Efeat.VANDERWAALS])
    assert edge_out_of_range.features[Efeat.VANDERWAALS] == 0.0, edge_out_of_range.features[
        Efeat.VANDERWAALS
    ]

    # MET 0 N - VAL 1 CA, intermediate distance, should have more negative
    # vanderwaals energy than the far interaction
    contact = AtomicContact(
        _get_atom(structure.chains[0], 0, "N"),
        _get_atom(structure.chains[0], 1, "CA"),
    )
    edge_intermediate = Edge(contact)
    add_features(pdb_path, _wrap_in_graph(edge_intermediate), variant)
    assert not np.isnan(edge_intermediate.features[Efeat.VANDERWAALS])
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
    assert not np.isnan(close_attracting_edge.features[Efeat.ELECTROSTATIC])
    assert (
        close_attracting_edge.features[Efeat.ELECTROSTATIC] < 0.0
    ), close_attracting_edge.features[Efeat.ELECTROSTATIC]

    # ARG 139 CZ - ASP 109 OD2, far attractive electrostatic energy
    contact = AtomicContact(
        _get_atom(structure.chains[0], 139, "CZ"),
        _get_atom(structure.chains[0], 109, "OE2"),
    )
    far_attracting_edge = Edge(contact)
    add_features(pdb_path, _wrap_in_graph(far_attracting_edge), variant)
    assert not np.isnan(far_attracting_edge.features[Efeat.ELECTROSTATIC])
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
    assert not np.isnan(opposing_edge.features[Efeat.ELECTROSTATIC])
    assert (
        opposing_edge.features[Efeat.ELECTROSTATIC] > 0.0
    ), opposing_edge.features[Efeat.ELECTROSTATIC]

    # check that we can calculate residue contacts
    contact = ResidueContact(
        structure.chains[0].residues[0], structure.chains[0].residues[1]
    )
    edge = Edge(contact)
    add_features(pdb_path, _wrap_in_graph(edge), variant)
    assert not np.isnan(edge.features[Efeat.DISTANCE]) > 0.0
    assert edge.features[Efeat.DISTANCE] > 0.0
    assert edge.features[Efeat.DISTANCE] < 1e5

    assert not np.isnan(edge.features[Efeat.ELECTROSTATIC])
    assert edge.features[Efeat.ELECTROSTATIC] != 0.0, edge.features[
        Efeat.ELECTROSTATIC
    ]

    assert not np.isnan(edge.features[Efeat.VANDERWAALS])
    assert edge.features[Efeat.VANDERWAALS] != 0.0, edge.features[
        Efeat.VANDERWAALS
    ]
