import os
from typing import Dict, Union

from pdb2sql import StructureSimilarity
from deeprankcore.domain import targettypes as targets


def get_all_scores(pdb_path: str, reference_pdb_path: str) -> Dict[str, Union[float, int]]:

    """Computes scores (lrmsd, irmsd, fnat, dockq, bin_class, capri_class) and outputs them as a dictionary

    Args:
        pdb_path (path): path to the scored pdb structure
        reference_pdb_path (path): path to the reference structure required to compute the different score

    Returns: a dictionary containing values for lrmsd, irmsd, fnat, dockq, bin_class, capri_class
    """

    ref_name = os.path.splitext(os.path.basename(reference_pdb_path))[0]
    sim = StructureSimilarity(pdb_path, reference_pdb_path)

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

    scores[targets.CAPRI] = 5
    for thr, val in zip([6.0, 4.0, 2.0, 1.0], [4, 3, 2, 1]):
        if scores[targets.IRMSD] < thr:
            scores[targets.CAPRI] = val

    return scores
