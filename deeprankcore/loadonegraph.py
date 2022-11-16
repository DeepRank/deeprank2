import logging
import warnings
import torch
import numpy as np
from torch_geometric.data.data import Data
import h5py
from deeprankcore.domain.features import groups
from deeprankcore.domain import targettypes as targets
# from typing import Callable, List, Union


_log = logging.getLogger(__name__)


def load_one_graph(fname, mol, 
                    transform = None,
                    target=None,
                    edge_features_transform = None,
                    clustering_method = None):
    """Loads one graph

    Args:
        fname (str): hdf5 file name
        mol (str): name of the molecule

    Returns:
        Data object or None: torch_geometric Data object containing the node features,
        the internal and external edge features, the target and the xyz coordinates.
        Return None if features cannot be loaded.
    """

    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with h5py.File(fname, 'r') as f5:
        grp = f5[mol]

        # node features
        node_data = ()
        for feat in  grp[f"{groups.NODE}"]:
            if feat[0] != '_':  # ignore metafeatures
                vals = grp[f"{groups.NODE}/{feat}"][()]
                if vals.ndim == 1:
                    vals = vals.reshape(-1, 1)
                node_data += (vals,)
        x = torch.tensor(np.hstack(node_data), dtype=torch.float).to(device)

        # edge index
        # (we have to have all the edges i.e : (i,j) and (j,i) )
        if groups.INDEX in grp[groups.EDGE]:
            ind = grp[f"{groups.EDGE}/{groups.INDEX}"][()]
            if ind.ndim == 2:
                ind = np.vstack((ind, np.flip(ind, 1))).T
            edge_index = torch.tensor(ind, dtype=torch.long).contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_index = edge_index.to(device)

        # edge feature
        # (same issue as above)
        if groups.EDGE in grp and grp[f"{groups.EDGE}"] is not None and len(grp[f"{groups.EDGE}"]) > 0:
            edge_data = ()
            for feat in grp[f"{groups.EDGE}"]:
                if feat[0] != '_':   # ignore metafeatures
                    vals = grp[f"{groups.EDGE}/{feat}"][()]
                    if vals.ndim == 1:
                        vals = vals.reshape(-1, 1)
                    edge_data += (vals,)
            edge_data = np.hstack(edge_data)
            edge_data = np.vstack((edge_data, edge_data))
            if edge_features_transform is not None:
                edge_data = edge_features_transform(edge_data)
            edge_attr = torch.tensor(edge_data, dtype=torch.float).contiguous()
        else:
            edge_attr = torch.empty((edge_index.shape[1], 0), dtype=torch.float).contiguous()
        edge_attr = edge_attr.to(device)

        if any(key in grp for key in ("internal_edge_index", "internal_edge_data")):
            warnings.warn(
                """Internal edges are not supported anymore.
                You should probably prepare the hdf5 file
                with a more up to date version of this software.""", DeprecationWarning)

        # target
        if target is None:
            y = None
        else:
            if targets.VALUES in grp and target in grp[targets.VALUES]:
                try:
                    y = torch.tensor([grp[f"{targets.VALUES}/{target}"][()]], dtype=torch.float).contiguous().to(device)
                except Exception as e: # be more specific about which exception we are trying to catch
                    _log.error(e)
                    _log.info('If your target variable contains categorical classes, \
                    please convert them into class indices before defining the GraphDataset instance.')
            else:
                possible_targets = grp[targets.VALUES].keys()
                raise ValueError(f"Target {target} not found in entry {mol} in file {fname}." \
                                 f"\nAvailable targets are {possible_targets}." \
                                  "\nUse the query class to add target values to input data.")

        # positions
        pos = torch.tensor(grp[f"{groups.NODE}/{groups.POSITION}/"][()], dtype=torch.float).contiguous().to(device)

        # cluster
        cluster0 = None
        cluster1 = None
        if clustering_method is not None:
            if 'clustering' in grp.keys():
                if clustering_method in grp["clustering"].keys():
                    if (
                        "depth_0" in grp[f"clustering/{clustering_method}"].keys() and
                        "depth_1" in grp[f"clustering/{clustering_method}"].keys()
                        ):

                        cluster0 = torch.tensor(
                            grp["clustering/" + clustering_method + "/depth_0"][()], dtype=torch.long).to(device)
                        cluster1 = torch.tensor(
                            grp["clustering/" + clustering_method + "/depth_1"][()], dtype=torch.long).to(device)
                    else:
                        _log.warning("no clusters detected")
                else:
                    _log.warning(f"no clustering/{clustering_method} detected")
            else:
                _log.warning("no clustering group found")
        # else:
        #     _log.warning("no cluster method set")

    # load
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)
    data.cluster0 = cluster0
    data.cluster1 = cluster1
    data.mol = mol

    # apply transformation
    if transform is not None:
        data = transform(data)

    return data