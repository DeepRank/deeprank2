import glob
import logging
import os

import h5py
import numpy as np
from pdb2sql import StructureSimilarity

from deeprank2.domain import targetstorage as targets

_log = logging.getLogger(__name__)
MIN_IRMS_FOR_BINARY = 4


def add_target(  # noqa: C901
    graph_path: str | list[str],
    target_name: str,
    target_list: str,
    sep: str = " ",
) -> None:
    """Add a target to all the graphs in hdf5 files.

    Args:
        graph_path: Either a directory containing all the hdf5 files, a single hdf5 filename, or a list of hdf5 filenames.
        target_name: The name of the new target.
        target_list: Name of the file containing the data.
        sep: Separator in target list. Defaults to " " (single space).

    Notes:
        The input target list should respect the following format :
        1ATN_xxx-1 0
        1ATN_xxx-2 1
        1ATN_xxx-3 0
        1ATN_xxx-4 0
    """
    labels = np.loadtxt(target_list, delimiter=sep, usecols=[0], dtype=str)
    values = np.loadtxt(target_list, delimiter=sep, usecols=[1])
    target_dict = dict(zip(labels, values, strict=False))

    if os.path.isdir(graph_path):
        graphs = glob.glob(f"{graph_path}/*.hdf5")
    elif os.path.isfile(graph_path):
        graphs = [graph_path]
    elif isinstance(graph_path, list):
        graphs = graph_path
    else:
        msg = "Incorrect input passed."
        raise TypeError(msg)

    for hdf5 in graphs:
        _log.info(hdf5)
        if not os.path.isfile(hdf5):
            msg = f"File {hdf5} not found."
            raise FileNotFoundError(msg)

        try:
            f5 = h5py.File(hdf5, "a")
            for model in target_dict:
                if model not in f5:
                    msg = f"{hdf5} does not contain an entry named {model}."
                    raise ValueError(msg)  # noqa: TRY301
                try:
                    model_gp = f5[model]
                    if targets.VALUES not in model_gp:
                        model_gp.create_group(targets.VALUES)
                    group = f5[f"{model}/{targets.VALUES}/"]
                    if target_name in group:
                        # Delete the target if it already existed
                        del group[target_name]
                    # Create the target
                    group.create_dataset(target_name, data=target_dict[model])
                except BaseException:  # noqa: BLE001
                    _log.info(f"no graph for {model}")
            f5.close()

        except BaseException:  # noqa: BLE001
            _log.info(f"no graph for {hdf5}")


def compute_ppi_scores(
    pdb_path: str,
    reference_pdb_path: str,
) -> dict[str, float | int]:
    """Compute structure similarity scores for the input docking model and return them as a dictionary.

    The computed scores are: `lrmsd` (ligand root mean square deviation), `irmsd` (interface rmsd),
    `fnat` (fraction of native contacts), `dockq` (docking model quality), `binary` (True - high quality,
    False - low quality), `capri_class` (capri classification, 1 - high quality, 2 - medium, 3 - acceptable,
    4 - incorrect). See https://deeprank2.readthedocs.io/en/latest/docking.html for more details about the scores.

    Args:
        pdb_path: Path to the decoy.
        reference_pdb_path: Path to the reference (native) structure.

    Returns: dict containing values for lrmsd, irmsd, fnat, dockq, binary, capri_class.
    """
    ref_name = os.path.splitext(os.path.basename(reference_pdb_path))[0]
    sim = StructureSimilarity(
        pdb_path,
        reference_pdb_path,
        enforce_residue_matching=False,
    )

    scores = {}

    # Input pre-computed zone files
    if os.path.exists(ref_name + ".lzone"):
        scores[targets.LRMSD] = sim.compute_lrmsd_fast(method="svd", lzone=ref_name + ".lzone")
        scores[targets.IRMSD] = sim.compute_irmsd_fast(method="svd", izone=ref_name + ".izone")

    # Compute zone files
    else:
        scores[targets.LRMSD] = sim.compute_lrmsd_fast(method="svd")
        scores[targets.IRMSD] = sim.compute_irmsd_fast(method="svd")

    scores[targets.FNAT] = sim.compute_fnat_fast()
    scores[targets.DOCKQ] = sim.compute_DockQScore(scores[targets.FNAT], scores[targets.LRMSD], scores[targets.IRMSD])
    scores[targets.BINARY] = scores[targets.IRMSD] < MIN_IRMS_FOR_BINARY

    scores[targets.CAPRI] = 4
    for thr, val in zip([4.0, 2.0, 1.0], [3, 2, 1], strict=True):
        if scores[targets.IRMSD] < thr:
            scores[targets.CAPRI] = val

    return scores
