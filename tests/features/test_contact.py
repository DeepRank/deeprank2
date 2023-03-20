from uuid import uuid4
from pdb2sql import pdb2sql
import numpy as np
from deeprankcore.molstruct.structure import Chain
from deeprankcore.molstruct.atom import Atom
from deeprankcore.molstruct.pair import AtomicContact, ResidueContact
from deeprankcore.molstruct.variant import SingleResidueVariant
from deeprankcore.utils.graph import Edge, Graph
from deeprankcore.utils.buildgraph import get_structure
from deeprankcore.features.contact import add_features
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


def _get_contact(pdb_id: str, residue_num1: int, atom_name1: str, residue_num2: int, atom_name2: str, residue_level: bool = False) -> Edge:
    pdb_path = f"tests/data/pdb/101M/{pdb_id}.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, pdb_id)
    finally:
        pdb._close() # pylint: disable=protected-access

    variant = SingleResidueVariant(structure.chains[0].residues[10], alanine)

    if not residue_level:
        contact = AtomicContact(
            _get_atom(structure.chains[0], residue_num1, atom_name1), 
            _get_atom(structure.chains[0], residue_num2, atom_name2)
        )
    else:
        contact = ResidueContact(
            structure.chains[0].residues[residue_num1], 
            structure.chains[0].residues[residue_num2]
        )

    edge_obj = Edge(contact)
    add_features(pdb_path, _wrap_in_graph(edge_obj), variant)
    
    assert not np.isnan(edge_obj.features[Efeat.VANDERWAALS]), 'isnan vdw'
    assert not np.isnan(edge_obj.features[Efeat.ELECTROSTATIC]), 'isnan electrostatic'
    assert not np.isnan(edge_obj.features[Efeat.DISTANCE]), 'isnan distance'
    assert not np.isnan(edge_obj.features[Efeat.SAMECHAIN]), 'isnan samechain'
    assert not np.isnan(edge_obj.features[Efeat.COVALENT]), 'isnan covalent'
    if not residue_level:
        assert not np.isnan(edge_obj.features[Efeat.SAMERES]), 'isnan sameres'

    return edge_obj


def test_covalent_pair():
    """MET 0: N - CA, covalent pair (at 1.49 A distance). Should have 0 vanderwaals and electrostatic energies.
    """

    edge_covalent = _get_contact('101M', 0, "N", 0, "CA")
    assert edge_covalent.features[Efeat.VANDERWAALS] == 0.0, 'nonzero vdw energy for covalent pair'
    assert edge_covalent.features[Efeat.ELECTROSTATIC] == 0.0, 'nonzero electrostatic energy for covalent pair'
    assert edge_covalent.features[Efeat.COVALENT] == 1.0, 'covalent pair not recognized as covalent'


def test_13_pair():
    """MET 0: N - CB, 1-3 pair (at 2.47 A distance). Should have 0 vanderwaals and electrostatic energies.
    """

    edge_13 = _get_contact('101M', 0, "N", 0, "CB")
    assert edge_13.features[Efeat.VANDERWAALS] == 0.0, 'nonzero vdw energy for 1-3 pair'
    assert edge_13.features[Efeat.ELECTROSTATIC] == 0.0, 'nonzero electrostatic energy for 1-3 pair'
    assert edge_13.features[Efeat.COVALENT] == 0.0, '1-3 pair recognized as covalent'
    

def test_14_pair():
    """MET 0: N - CG, 1-4 pair (at 4.12 A distance). Should have non-zero electrostatic energy and small non-zero vdw energy.
    """

    edge_14 = _get_contact('101M', 0, "CA", 0, "SD")
    assert edge_14.features[Efeat.VANDERWAALS] != 0.0, '1-4 pair with 0 vdw energy'
    assert abs(edge_14.features[Efeat.VANDERWAALS]) < 0.1, '1-4 pair with large vdw energy'
    assert edge_14.features[Efeat.ELECTROSTATIC] != 0.0, '1-4 pair with 0 electrostatic'
    assert edge_14.features[Efeat.COVALENT] == 0.0, '1-4 pair recognized as covalent'


def test_vanderwaals_negative():
    """MET 0 N - ASP 27 CB, very far (29.54 A). Should have negative vanderwaals energy.
    """

    edge_far = _get_contact('101M', 0, "N", 27, "CB")
    assert edge_far.features[Efeat.VANDERWAALS] < 0.0

    
def test_vanderwaals_morenegative():
    """MET 0 N - PHE 138 CG, intermediate distance (12.69 A). Should have more negative vanderwaals energy than the far interaction.
    """

    edge_intermediate = _get_contact('101M', 0, "N", 138, "CG")
    edge_far = _get_contact('101M', 0, "N", 27, "CB")
    assert edge_intermediate.features[Efeat.VANDERWAALS] < edge_far.features[Efeat.VANDERWAALS]


def test_edge_distance():
    """Check the edge distances.
    """

    edge_close = _get_contact('101M', 0, "N", 0, "CA")
    edge_intermediate = _get_contact('101M', 0, "N", 138, "CG")
    edge_far = _get_contact('101M', 0, "N", 27, "CB")

    assert (
        edge_close.features[Efeat.DISTANCE]
        < edge_intermediate.features[Efeat.DISTANCE]
    ), 'close distance > intermediate distance'
    assert (
        edge_far.features[Efeat.DISTANCE]
        > edge_intermediate.features[Efeat.DISTANCE]
    ), 'far distance < intermediate distance'


def test_attractive_electrostatic_close():
    """ARG 139 CZ - GLU 136 OE2, very close (5.60 A). Should have attractive electrostatic energy.
    """

    close_attracting_edge = _get_contact('101M', 139, "CZ", 136, "OE2")
    assert close_attracting_edge.features[Efeat.ELECTROSTATIC] < 0.0


def test_attractive_electrostatic_far():
    """ARG 139 CZ - ASP 20 OD2, far (24.26 A). Should have attractive more electrostatic energy than above.
    """

    far_attracting_edge = _get_contact('101M', 139, "CZ", 20, "OD2")
    close_attracting_edge = _get_contact('101M', 139, "CZ", 136, "OE2")
    assert (
        far_attracting_edge.features[Efeat.ELECTROSTATIC] < 0.0
    ), 'far electrostatic > 0'
    assert (
        far_attracting_edge.features[Efeat.ELECTROSTATIC]
        > close_attracting_edge.features[Efeat.ELECTROSTATIC]
    ), 'far electrostatic <= close electrostatic'
   

def test_repulsive_electrostatic():
    """GLU 109 OE2 - GLU 105 OE1 (9.64 A). Should have repulsive electrostatic energy.
    """

    opposing_edge = _get_contact('101M', 109, "OE2", 105, "OE1")
    assert opposing_edge.features[Efeat.ELECTROSTATIC] > 0.0


def test_residue_contact():
    """Check that we can calculate features for residue contacts.
    """

    res_edge = _get_contact('101M', 0, '', 1, '', residue_level = True)
    assert res_edge.features[Efeat.DISTANCE] > 0.0, 'distance <= 0'
    assert res_edge.features[Efeat.DISTANCE] < 1e5, 'distance > 1e5'
    assert res_edge.features[Efeat.ELECTROSTATIC] != 0.0, 'electrostatic == 0'
    assert res_edge.features[Efeat.VANDERWAALS] != 0.0, 'vanderwaals == 0'
    assert res_edge.features[Efeat.COVALENT] == 1.0, 'neighboring residues not seen as covalent'
