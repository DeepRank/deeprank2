import logging
import numpy as np
import warnings
from Bio.PDB.Atom import PDBConstructionWarning
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.ResidueDepth import get_surface, residue_depth
from Bio.PDB.HSExposure import HSExposureCA
from deeprankcore.domain import nodestorage as Nfeat
from deeprankcore.molstruct.atom import Atom
from deeprankcore.molstruct.residue import Residue
from deeprankcore.utils.graph import Graph
import sys
import signal


_log = logging.getLogger(__name__)


def handle_sigint(sig, frame):
    print('SIGINT received, terminating.')
    sys.exit()


def handle_timeout(sig, frame):
    raise TimeoutError(f'Bio.PDB.ResidueDepth.get_surface timed out')


def space_if_none(value):
    if value is None:
        return " "

    return value


def add_features(pdb_path: str, graph: Graph, *args, **kwargs): # pylint: disable=unused-argument

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
    except TimeoutError:
        raise
    else:
        hse = HSExposureCA(bio_model)

        for node in graph.nodes:

            # These can only be calculated per residue, not per atom.
            # So for atomic graphs, every atom gets its residue's value.
            if isinstance(node.id, Atom):
                atom = node.id
                residue = atom.residue
            elif isinstance(node.id, Residue):
                residue = node.id
            else:
                raise TypeError(f"Unexpected node type: {type(node)}")

            bio_residue = bio_model[residue.chain.id][residue.number]
            node.features[Nfeat.RESDEPTH] = residue_depth(bio_residue, surface)

            hse_key = (residue.chain.id, (" ", residue.number, space_if_none(residue.insertion_code)))
            if hse_key in hse:
                node.features[Nfeat.HSE] = hse[hse_key]
            else:
                node.features[Nfeat.HSE] = np.array((0, 0, 0))
