import glob
import os
from typing import Dict, List, Union

import h5py
import numpy as np
from pdb2sql import StructureSimilarity

from deeprank2.domain import targetstorage as targets


def add_target(graph_path: Union[str, List[str]], target_name: str, target_list: str, sep: str = " "):
    """Add a target to all the graphs in hdf5 files.

    Args:
        graph_path (Union[str, List(str)]): Either a directory containing all the hdf5 files,
            or a single hdf5 filename
            or a list of hdf5 filenames.
        target_name (str): The name of the new target.
        target_list (str): Name of the file containing the data.
        sep (str, optional): Separator in target list. Defaults to " ".

    Notes:
        The input target list should respect the following format :
        1ATN_xxx-1 0
        1ATN_xxx-2 1
        1ATN_xxx-3 0
        1ATN_xxx-4 0
    """

    target_dict = {}

    labels = np.loadtxt(target_list, delimiter=sep, usecols=[0], dtype=str)
    values = np.loadtxt(target_list, delimiter=sep, usecols=[1])
    for label, value in zip(labels, values):
        target_dict[label] = value

    # if a directory is provided
    if os.path.isdir(graph_path):
        graphs = glob.glob(f"{graph_path}/*.hdf5")

    # if a single file is provided
    elif os.path.isfile(graph_path):
        graphs = [graph_path]

    # if a list of file is provided
    else:
        assert isinstance(graph_path, list)
        assert os.path.isfile(graph_path[0])

    for hdf5 in graphs:
        print(hdf5)
        try:
            f5 = h5py.File(hdf5, "a")

            for model, _ in target_dict.items():
                if model not in f5:
                    raise ValueError(
                        f"{hdf5} does not contain an entry named {model}"
                    )

                try:
                    model_gp = f5[model]

                    if targets.VALUES not in model_gp:
                        model_gp.create_group(targets.VALUES)

                    group = f5[f"{model}/{targets.VALUES}/"]

                    if target_name in group.keys():
                        # Delete the target if it already existed
                        del group[target_name]

                    # Create the target
                    group.create_dataset(target_name, data=target_dict[model])

                except BaseException:
                    print(f"no graph for {model}")

            f5.close()

        except BaseException:
            print(f"no graph for {hdf5}")


def compute_ppi_scores(pdb_path: str, reference_pdb_path: str) -> Dict[str, Union[float, int]]:

    """Compute structure similarity scores for the input docking model and return them as a dictionary.

    The computed scores are: `lrmsd` (ligand root mean square deviation), `irmsd` (interface rmsd),
    `fnat` (fraction of native contacts), `dockq` (docking model quality), `binary` (True - high quality,
    False - low quality), `capri_class` (capri classification, 1 - high quality, 2 - medium, 3 - acceptable,
    4 - incorrect). See https://deeprank2.readthedocs.io/en/latest/docking.html for more details about the scores.

    Args:
        pdb_path (str): Path to the decoy.
        reference_pdb_path (str): Path to the reference (native) structure.

    Returns: a dictionary containing values for lrmsd, irmsd, fnat, dockq, binary, capri_class.
    """

    ref_name = os.path.splitext(os.path.basename(reference_pdb_path))[0]
    sim = StructureSimilarity(pdb_path, reference_pdb_path, enforce_residue_matching=False)

    scores = {}

    # Input pre-computed zone files
    if os.path.exists(ref_name + ".lzone"):
        scores[targets.LRMSD] = sim.compute_lrmsd_fast(
            method="svd", lzone=ref_name + ".lzone"
        )
        scores[targets.IRMSD] = sim.compute_irmsd_fast(
            method="svd", izone=ref_name + ".izone"
        )

    # Compute zone files
    else:
        scores[targets.LRMSD] = sim.compute_lrmsd_fast(method="svd")
        scores[targets.IRMSD] = sim.compute_irmsd_fast(method="svd")

    scores[targets.FNAT] = sim.compute_fnat_fast()
    scores[targets.DOCKQ] = sim.compute_DockQScore(
        scores[targets.FNAT], scores[targets.LRMSD], scores[targets.IRMSD]
    )
    scores[targets.BINARY] = scores[targets.IRMSD] < 4.0

    scores[targets.CAPRI] = 4
    for thr, val in zip([4.0, 2.0, 1.0], [3, 2, 1]):
        if scores[targets.IRMSD] < thr:
            scores[targets.CAPRI] = val

    return scores
