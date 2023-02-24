from uuid import uuid4
from pdb2sql import pdb2sql
import numpy as np
from typing import List
from scipy.spatial import distance_matrix
from deeprankcore.molstruct.structure import Chain, PDBStructure
from deeprankcore.molstruct.atom import Atom
from deeprankcore.molstruct.pair import AtomicContact, ResidueContact
from deeprankcore.molstruct.variant import SingleResidueVariant
from deeprankcore.utils.graph import Edge, Graph
from deeprankcore.utils.buildgraph import get_structure
from deeprankcore.features.contact import add_features, _intra_partners
from deeprankcore.domain.aminoacidlist import alanine
from deeprankcore.domain import edgestorage as Efeat
from tests.features.test_contact import _get_atom


def _abs_distance(positions, atom1: List[int], atom2: List[int]):
    pos1, pos2 = positions[atom1], positions[atom2]
    sum_of_squares = 0
    for i in range(3):
        sum_of_squares += (pos1[i]-pos2[i])**2
    return np.sqrt(sum_of_squares)


def test_within_3bonds_distinction():
    pdb = pdb2sql("tests/data/pdb/1ak4/1ak4.pdb")
    try:
        structure = get_structure(pdb, "101m")
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
    index_D_pro93_CD = atoms.index(_get_atom(chain_D, 93, "CD"))
    index_D_ala92_CA = atoms.index(_get_atom(chain_D, 92, "CA"))
    index_D_ala92_CB = atoms.index(_get_atom(chain_D, 92, "CB"))
    index_D_gly89_N = atoms.index(_get_atom(chain_D, 89, "N"))

    # one bond away
    print('1', _abs_distance(positions, index_D_pro93_CA, index_D_pro93_CB))
    assert intra_matrix[index_D_pro93_CA, index_D_pro93_CB]
    assert intra_matrix[index_D_pro93_CB, index_D_pro93_CA]

    # two bonds away
    print('2', _abs_distance(positions, index_D_pro93_CA, index_D_pro93_CG))
    assert intra_matrix[index_D_pro93_CA, index_D_pro93_CG]
    assert intra_matrix[index_D_pro93_CG, index_D_pro93_CA]

    # three bonds away
    print('3', _abs_distance(positions, index_D_pro93_CA, index_D_ala92_CA))
    assert intra_matrix[index_D_pro93_CA, index_D_ala92_CA]
    assert intra_matrix[index_D_ala92_CA, index_D_pro93_CA]

    # four bonds away
    print('4', _abs_distance(positions, index_D_pro93_CA, index_D_ala92_CB))
    assert not intra_matrix[index_D_pro93_CA, index_D_ala92_CB]

    # in different chain, but hydrogen bonded
    print('Hbond, diff chain', _abs_distance(positions, index_D_gly89_N, index_C_asn102_O))
    assert not intra_matrix[index_D_gly89_N, index_C_asn102_O]

    # close, but not connected
    print('close, but not connected', _abs_distance(positions, index_C_trp121_CZ2, index_C_phe60_CE1))
    assert not intra_matrix[index_C_trp121_CZ2, index_C_phe60_CE1]

    # far away from each other
    print('far', _abs_distance(positions, index_D_leu111_CG, index_D_pro93_CA))
    assert not intra_matrix[index_D_leu111_CG, index_D_pro93_CA]

    assert True==False, "no other test failed"