import logging
import signal
import sys
import warnings
from typing import NoReturn

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


def handle_sigint(sig, frame) -> None:  # noqa: ARG001, ANN001, D103
    _log.info("SIGINT received, terminating.")
    sys.exit()


def handle_timeout(sig, frame) -> NoReturn:  # noqa: ARG001, ANN001, D103
    msg = "Timed out!"
    raise TimeoutError(msg)


def space_if_none(value: str) -> str:  # noqa:D103
    if value is None:
        return " "
    return value


def add_features(  # noqa:D103
    pdb_path: str,
    graph: Graph,
    single_amino_acid_variant: SingleResidueVariant | None = None,  # noqa: ARG001
) -> None:
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGALRM, handle_timeout)

    with warnings.catch_warnings(record=PDBConstructionWarning):
        parser = PDBParser()
        structure = parser.get_structure("_tmp", pdb_path)
    bio_model = structure[0]

    try:
        signal.alarm(20)
        surface = get_surface(bio_model)
        signal.alarm(0)
    except TimeoutError as e:
        msg = "Bio.PDB.ResidueDepth.get_surface timed out."
        raise TimeoutError(msg) from e

    # These can only be calculated per residue, not per atom.
    # So for atomic graphs, every atom gets its residue's value.
    hse = HSExposureCA(bio_model)
    for node in graph.nodes:
        if isinstance(node.id, Residue):
            residue = node.id
        elif isinstance(node.id, Atom):
            atom = node.id
            residue = atom.residue
        else:
            msg = f"Unexpected node type: {type(node.id)}"
            raise TypeError(msg)

        bio_residue = bio_model[residue.chain.id][residue.number]
        node.features[Nfeat.RESDEPTH] = residue_depth(bio_residue, surface)
        hse_key = (
            residue.chain.id,
            (" ", residue.number, space_if_none(residue.insertion_code)),
        )

        if hse_key in hse:
            node.features[Nfeat.HSE] = np.array(hse[hse_key], dtype=np.float64)
        else:
            node.features[Nfeat.HSE] = np.array((0, 0, 0), dtype=np.float64)
