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

    """
    Compute structure similarity scores and return them as a dictionary.
    Such measures have been developed for evaluating the quality of the PPI models produced by
    computational methods (docking models), and all of them compare the structural similarity
    between the decoys (computationally generated structures) and the experimentally solved native
    structures. To calculate these measures, the interface between the two interacting protein molecules
    is defined as any pair of heavy atoms from the two molecules within 5Å of each other.
    For regression:
       - ligand root mean square deviation (lrmsd), float. It is calculated for the backbone of
       the shorter chain (ligand) of the model after superposition of the longer chain (receptor).
       The lower the better.
       - interface rmsd (irmsd), float. The backbone atoms of the interface residues (atomic contact cutoff
       of 10Å) is superposed on their equivalents in the predicted complex (model) to compute it.
       The lower the better.
       - fraction of native contacts (fnat), float. The fraction of native interfacial contacts preserved in
       the interface of the predicted complex. The score is in the range [0, 1], corresponding to low and
       high quality, respectively.
       - dockq, float. It is a continuous quality measure for docking models that instead of classifying into different
       quality groups, combines Fnat, LRMS, and iRMS to yield a score in the range [0, 1], corresponding to low and
       high quality, respectively.
    For classification:
       - binary (bool). True if the irmsd is lower than 4.0, meaning that the decoy is considered high quality
       docking model, otherwise False.
       - capri_classes (int). The possible values are: 4 (incorrect), 3 (acceptable), 2 (medium), 1 (high quality).
    See https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.10393
    for more details about capri_classes, lrmsd, irmsd, and fnat.
    See https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0161879
    for more details about dockq.

    Args:
        pdb_path (str): Path to the decoy.
        reference_pdb_path (str): Path to the reference (native) structure.

    Returns: a dictionary containing values for lrmsd, irmsd, fnat, dockq, binary, capri_class
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
    for thr, val in zip([6.0, 4.0, 2.0, 1.0], [4, 3, 2, 1]):
        if scores[targets.IRMSD] < thr:
            scores[targets.CAPRI] = val

    return scores
