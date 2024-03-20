from uuid import uuid4

import numpy as np
from pdb2sql import pdb2sql

from deeprank2.domain import edgestorage as Efeat
from deeprank2.features.contact import add_features, covalent_cutoff, cutoff_13, cutoff_14
from deeprank2.molstruct.atom import Atom
from deeprank2.molstruct.pair import AtomicContact, ResidueContact
from deeprank2.molstruct.structure import Chain
from deeprank2.utils.buildgraph import get_structure
from deeprank2.utils.graph import Edge, Graph


def _get_atom(chain: Chain, residue_number: int, atom_name: str) -> Atom:
    for residue in chain.residues:
        if residue.number == residue_number:
            for atom in residue.atoms:
                if atom.name == atom_name:
                    return atom
    msg = f"Not found: chain {chain.id} residue {residue_number} atom {atom_name}"
    raise ValueError(msg)


def _wrap_in_graph(edge: Edge) -> Graph:
    g = Graph(uuid4().hex)
    g.add_edge(edge)
    return g


def _get_contact(
    pdb_id: str,
    residue_num1: int,
    atom_name1: str,
    residue_num2: int,
    atom_name2: str,
    residue_level: bool = False,
    chains: tuple[str, str] | None = None,
) -> Edge:
    pdb_path = f"tests/data/pdb/{pdb_id}/{pdb_id}.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, pdb_id)
    finally:
        pdb._close()

    if not chains:
        chains = [structure.chains[0], structure.chains[0]]
    else:
        chains = [structure.get_chain(chain) for chain in chains]

    if not residue_level:
        contact = AtomicContact(
            _get_atom(chains[0], residue_num1, atom_name1),
            _get_atom(chains[1], residue_num2, atom_name2),
        )
    else:
        contact = ResidueContact(chains[0].residues[residue_num1], chains[1].residues[residue_num2])

    edge_obj = Edge(contact)
    add_features(pdb_path, _wrap_in_graph(edge_obj))

    assert not np.isnan(edge_obj.features[Efeat.VDW]), "isnan vdw"
    assert not np.isnan(edge_obj.features[Efeat.ELEC]), "isnan electrostatic"
    assert not np.isnan(edge_obj.features[Efeat.DISTANCE]), "isnan distance"
    assert not np.isnan(edge_obj.features[Efeat.SAMECHAIN]), "isnan samechain"
    assert not np.isnan(edge_obj.features[Efeat.COVALENT]), "isnan covalent"
    if not residue_level:
        assert not np.isnan(edge_obj.features[Efeat.SAMERES]), "isnan sameres"

    return edge_obj


def test_covalent_pair() -> None:
    """MET 0: N - CA, covalent pair (at 1.49 A distance). Should have 0 vanderwaals and electrostatic energies."""
    edge_covalent = _get_contact("101M", 0, "N", 0, "CA")
    assert edge_covalent.features[Efeat.DISTANCE] < covalent_cutoff
    assert edge_covalent.features[Efeat.VDW] == 0.0, "nonzero vdw energy for covalent pair"
    assert edge_covalent.features[Efeat.ELEC] == 0.0, "nonzero electrostatic energy for covalent pair"
    assert edge_covalent.features[Efeat.COVALENT] == 1.0, "covalent pair not recognized as covalent"


def test_13_pair() -> None:
    """MET 0: N - CB, 1-3 pair (at 2.47 A distance). Should have 0 vanderwaals and electrostatic energies."""
    edge_13 = _get_contact("101M", 0, "N", 0, "CB")
    assert edge_13.features[Efeat.DISTANCE] < cutoff_13
    assert edge_13.features[Efeat.VDW] == 0.0, "nonzero vdw energy for 1-3 pair"
    assert edge_13.features[Efeat.ELEC] == 0.0, "nonzero electrostatic energy for 1-3 pair"
    assert edge_13.features[Efeat.COVALENT] == 0.0, "1-3 pair recognized as covalent"


def test_very_close_opposing_chains() -> None:
    """ChainA THR 118 O - ChainB ARG 30 NH1 (3.55 A). Should have non-zero energy despite close contact, because opposing chains."""
    opposing_edge = _get_contact("1A0Z", 118, "O", 30, "NH1", chains=("A", "B"))
    assert opposing_edge.features[Efeat.DISTANCE] < cutoff_13
    assert opposing_edge.features[Efeat.ELEC] != 0.0
    assert opposing_edge.features[Efeat.VDW] != 0.0


def test_14_pair() -> None:
    """MET 0: N - CG, 1-4 pair (at 4.12 A distance). Should have non-zero electrostatic energy and small non-zero vdw energy."""
    edge_14 = _get_contact("101M", 0, "CA", 0, "SD")
    assert edge_14.features[Efeat.DISTANCE] > cutoff_13
    assert edge_14.features[Efeat.DISTANCE] < cutoff_14
    assert edge_14.features[Efeat.VDW] != 0.0, "1-4 pair with 0 vdw energy"
    assert abs(edge_14.features[Efeat.VDW]) < 0.1, "1-4 pair with large vdw energy"
    assert edge_14.features[Efeat.ELEC] != 0.0, "1-4 pair with 0 electrostatic"
    assert edge_14.features[Efeat.COVALENT] == 0.0, "1-4 pair recognized as covalent"


def test_14dist_opposing_chains() -> None:
    """ChainA PRO 114 CA - ChainB HIS 116 CD2 (3.62 A).

    Should have non-zero energy despite close contact, because opposing chains.
    E_vdw for this pair if they were on the same chain: 0.018
    E_vdw for this pair on opposing chains: 0.146.
    """
    opposing_edge = _get_contact("1A0Z", 114, "CA", 116, "CD2", chains=("A", "B"))
    assert opposing_edge.features[Efeat.DISTANCE] > cutoff_13
    assert opposing_edge.features[Efeat.DISTANCE] < cutoff_14
    assert opposing_edge.features[Efeat.ELEC] > 1.0, f"electrostatic: {opposing_edge.features[Efeat.ELEC]}"
    assert opposing_edge.features[Efeat.VDW] > 0.1, f"vdw: {opposing_edge.features[Efeat.VDW]}"


def test_vanderwaals_negative() -> None:
    """MET 0 N - ASP 27 CB, very far (29.54 A). Should have negative vanderwaals energy."""
    edge_far = _get_contact("101M", 0, "N", 27, "CB")
    assert edge_far.features[Efeat.VDW] < 0.0


def test_vanderwaals_morenegative() -> None:
    """MET 0 N - PHE 138 CG, intermediate distance (12.69 A). Should have more negative vanderwaals energy than the far interaction."""
    edge_intermediate = _get_contact("101M", 0, "N", 138, "CG")
    edge_far = _get_contact("101M", 0, "N", 27, "CB")
    assert edge_intermediate.features[Efeat.VDW] < edge_far.features[Efeat.VDW]


def test_edge_distance() -> None:
    """Check the edge distances."""
    edge_close = _get_contact("101M", 0, "N", 0, "CA")
    edge_intermediate = _get_contact("101M", 0, "N", 138, "CG")
    edge_far = _get_contact("101M", 0, "N", 27, "CB")

    assert edge_close.features[Efeat.DISTANCE] < edge_intermediate.features[Efeat.DISTANCE], "close distance > intermediate distance"
    assert edge_far.features[Efeat.DISTANCE] > edge_intermediate.features[Efeat.DISTANCE], "far distance < intermediate distance"


def test_attractive_electrostatic_close() -> None:
    """ARG 139 CZ - GLU 136 OE2, very close (5.60 A). Should have attractive electrostatic energy."""
    close_attracting_edge = _get_contact("101M", 139, "CZ", 136, "OE2")
    assert close_attracting_edge.features[Efeat.ELEC] < 0.0


def test_attractive_electrostatic_far() -> None:
    """ARG 139 CZ - ASP 20 OD2, far (24.26 A). Should have attractive more electrostatic energy than above."""
    far_attracting_edge = _get_contact("101M", 139, "CZ", 20, "OD2")
    close_attracting_edge = _get_contact("101M", 139, "CZ", 136, "OE2")
    assert far_attracting_edge.features[Efeat.ELEC] < 0.0, "far electrostatic > 0"
    assert far_attracting_edge.features[Efeat.ELEC] > close_attracting_edge.features[Efeat.ELEC], "far electrostatic <= close electrostatic"


def test_repulsive_electrostatic() -> None:
    """GLU 109 OE2 - GLU 105 OE1 (9.64 A). Should have repulsive electrostatic energy."""
    opposing_edge = _get_contact("101M", 109, "OE2", 105, "OE1")
    assert opposing_edge.features[Efeat.ELEC] > 0.0


def test_residue_contact() -> None:
    """Check that we can calculate features for residue contacts."""
    res_edge = _get_contact("101M", 0, "", 1, "", residue_level=True)
    assert res_edge.features[Efeat.DISTANCE] > 0.0, "distance <= 0"
    assert res_edge.features[Efeat.DISTANCE] < 1e5, "distance > 1e5"
    assert res_edge.features[Efeat.ELEC] != 0.0, "electrostatic == 0"
    assert res_edge.features[Efeat.VDW] != 0.0, "vanderwaals == 0"
    assert res_edge.features[Efeat.COVALENT] == 1.0, "neighboring residues not seen as covalent"
