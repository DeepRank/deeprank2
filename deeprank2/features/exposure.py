import logging
import signal
import sys
import warnings
from typing import Optional

import numpy as np
from Bio.PDB.Atom import PDBConstructionWarning
from Bio.PDB.HSExposure import HSExposureCA
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.ResidueDepth import get_surface, residue_depth

from deeprank2.domain import nodestorage as Nfeat
from deeprank2.molstruct.atom import Atom
from deeprank2.molstruct.residue import Residue, SingleResidueVariant
from deeprank2.utils.graph import Graph

_log = logging.getLogger(__name__)


def handle_sigint(sig, frame): # pylint: disable=unused-argument
    print('SIGINT received, terminating.')
    sys.exit()


def handle_timeout(sig, frame):
    raise TimeoutError('Timed out!')


def space_if_none(value):
    if value is None:
        return " "
    return value


def add_features( # pylint: disable=unused-argument
    pdb_path: str, graph: Graph,
    single_amino_acid_variant: Optional[SingleResidueVariant] = None
    ):

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGALRM, handle_timeout)

    with warnings.catch_warnings(record=PDBConstructionWarning):
        parser = PDBParser()
        structure = parser.get_structure('_tmp', pdb_path)
    bio_model = structure[0]

    try:
        signal.alarm(20)
        surface = get_surface(bio_model)
        signal.alarm(0)
    except TimeoutError as e:
        raise TimeoutError('Bio.PDB.ResidueDepth.get_surface timed out.') from e
    else:
        hse = HSExposureCA(bio_model)

        # These can only be calculated per residue, not per atom.
        # So for atomic graphs, every atom gets its residue's value.
        for node in graph.nodes:
            if isinstance(node.id, Residue):
                residue = node.id
            elif isinstance(node.id, Atom):
                atom = node.id
                residue = atom.residue
            else:
                raise TypeError(f"Unexpected node type: {type(node.id)}")

            bio_residue = bio_model[residue.chain.id][residue.number]
            node.features[Nfeat.RESDEPTH] = residue_depth(bio_residue, surface)
            hse_key = (residue.chain.id, (" ", residue.number, space_if_none(residue.insertion_code)))

            if hse_key in hse:
                node.features[Nfeat.HSE] = np.array(hse[hse_key], dtype=np.float64)
            else:
                node.features[Nfeat.HSE] = np.array((0, 0, 0), dtype=np.float64)
