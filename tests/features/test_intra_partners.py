from pdb2sql import pdb2sql
import numpy as np
from typing import List
from scipy.spatial import distance_matrix
from deeprankcore.utils.buildgraph import get_structure
from deeprankcore.features.contact import _intra_partners
from tests.features.test_contact import _get_atom


def _abs_distance(positions, atom1: List[int], atom2: List[int]):
    pos1, pos2 = positions[atom1], positions[atom2]
    sum_of_squares = 0
    for i in range(3):
        sum_of_squares += (pos1[i]-pos2[i])**2
    return np.sqrt(sum_of_squares)


def test_intra_partners():
    pdb = pdb2sql("tests/data/pdb/1ak4/1ak4.pdb")
    try:
        structure = get_structure(pdb, "1ak4")
    finally:
        pdb._close() # pylint: disable=protected-access

    chain_C = structure.get_chain('C')
    chain_D = structure.get_chain('D')

    atoms = structure.get_atoms()
    positions = np.array([atom.position for atom in atoms])
    distances = distance_matrix(positions, positions)

    count_atoms = len(atoms)

    intra_matrix = _intra_partners(distances, 3)

    assert intra_matrix.shape == (count_atoms, count_atoms)

    index_C_phe60_CE1 = atoms.index(_get_atom(chain_C, 60, "CE1"))
    index_C_trp121_CZ2 = atoms.index(_get_atom(chain_C, 121, "CZ2"))
    index_C_asn102_O = atoms.index(_get_atom(chain_C, 102, "O"))
    index_D_leu111_CG = atoms.index(_get_atom(chain_D, 111, "CG"))
    index_D_pro93_CA = atoms.index(_get_atom(chain_D, 93, "CA"))
    index_D_pro93_CB = atoms.index(_get_atom(chain_D, 93, "CB"))
    index_D_pro93_CG = atoms.index(_get_atom(chain_D, 93, "CG"))
    index_D_ala92_CA = atoms.index(_get_atom(chain_D, 92, "CA"))
    index_D_ala92_CB = atoms.index(_get_atom(chain_D, 92, "CB"))
    index_D_gly89_N = atoms.index(_get_atom(chain_D, 89, "N"))

    # one bond away
    dist = _abs_distance(positions, index_D_pro93_CA, index_D_pro93_CB)
    assert intra_matrix[index_D_pro93_CA, index_D_pro93_CB], f'1-2 pairing not considered intra; distance == {dist}'
    assert intra_matrix[index_D_pro93_CB, index_D_pro93_CA], '1-2 pairing, inverted'

    # two bonds away
    dist = _abs_distance(positions, index_D_pro93_CA, index_D_pro93_CG)
    assert intra_matrix[index_D_pro93_CA, index_D_pro93_CG], f'1-3 paring not considered intra; distance == {dist}'
    assert intra_matrix[index_D_pro93_CG, index_D_pro93_CA], '1-3 pairing, inverted'

    # three bonds away
    dist = _abs_distance(positions, index_D_pro93_CA, index_D_ala92_CA)
    assert intra_matrix[index_D_pro93_CA, index_D_ala92_CA], f'1-4 pairing not considered intra; distance == {dist}'
    assert intra_matrix[index_D_ala92_CA, index_D_pro93_CA], '1-4 pairing inverted'

    # four bonds away
    dist = _abs_distance(positions, index_D_pro93_CA, index_D_ala92_CB)
    assert not intra_matrix[index_D_pro93_CA, index_D_ala92_CB], f'1-5 pairing is considered intra; distance == {dist}'
    assert not intra_matrix[index_D_ala92_CB, index_D_pro93_CA], '1-5 pairing inverted'

    # far away from each other
    dist = _abs_distance(positions, index_D_leu111_CG, index_D_pro93_CA)
    assert not intra_matrix[index_D_leu111_CG, index_D_pro93_CA], f'far is considered intra; distance == {dist}'

    # in different chain, but hydrogen bonded
    dist = _abs_distance(positions, index_D_gly89_N, index_C_asn102_O)
    assert not intra_matrix[index_D_gly89_N, index_C_asn102_O], f'close, H-bonded separate chains is considered intra; distance == {dist}'

    # close, but not connected
    dist = _abs_distance(positions, index_C_trp121_CZ2, index_C_phe60_CE1)
    assert not intra_matrix[index_C_trp121_CZ2, index_C_phe60_CE1], f'close but not connected is considered intra; distance == {dist}'

    # very close connections are not always intra
    max_dist = 3.0
    very_close_by = distances < max_dist
    not_intra = np.logical_not(intra_matrix)
    assert np.any(np.logical_and(very_close_by, not_intra)), f'all connections within {dist} A are considered intra'
     