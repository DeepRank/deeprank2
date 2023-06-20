import torch as t
import numpy as np
from deeprankcore.utils.graph import Graph
from pdb2sql import pdb2sql
from matplotlib import pyplot
from deeprankcore.domain import nodestorage as Nfeat
import h5py

def add_features(
    pdb_path: str,
    graph: Graph,
):
    db = pdb2sql(pdb_path)
    chains = graph.get_all_chains()
    backbones = {chain: db.get("resSeq, name, x,y,z", name=["N", "CA", "C"], chainID=[chain]) for chain in chains}

    for node in graph.nodes:
        chain, resSeq = node.id._chain._id, node.id._number
        
        N = t.tensor([row[2:] for row in backbones[chain] if row[:2] == [resSeq, "N"]][0])
        N_plus1 = [row[2:] for row in backbones[chain] if row[:2] == [resSeq+1, "N"]]
        N_plus1 = t.tensor(N_plus1[0]) if len(N_plus1) == 1 else None

        Ca = t.tensor([row[2:] for row in backbones[chain] if row[:2] == [resSeq, "CA"]][0])
        Ca_plus1 = [row[2:] for row in backbones[chain] if row[:2] == [resSeq+1, "CA"]]
        Ca_plus1 = t.tensor(Ca_plus1[0]) if len(Ca_plus1) == 1 else None

        C = t.tensor([row[2:] for row in backbones[chain] if row[:2] == [resSeq, "C"]][0])
        C_minus1 = [row[2:] for row in backbones[chain] if row[:2] == [resSeq-1, "C"]]
        C_minus1 = t.tensor(C_minus1[0]) if len(C_minus1) == 1 else None

        psy = dihedral((N, Ca, C, N_plus1)) if N_plus1 is not None else 0
        omega = dihedral((Ca, C, N_plus1, Ca_plus1)) if N_plus1 is not None else 0
        phi = dihedral((C_minus1, N, Ca, C)) if C_minus1 is not None else 0

        torsion_angle = [phi, psy, omega]
        node.features[Nfeat.TORSIONANGLE] = torsion_angle
    pass

def dihedral(p):
    """Get dihedral angle from four points (atoms) 3D coordinates.

    Args:
        p (list): list of four points 3D coordinates.
                  Each element must be a list of floats.

    Returns:
        dihedral (float): Dihedral angle.
        """
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    dihedral = np.degrees(np.arctan2(y, x))
    return dihedral