import h5py
import pandas as pd
import logging
from typing import List, Union
from deeprankcore.domain import (
    edgestorage as Efeat,
    nodestorage as Nfeat,
    targetstorage as targets)


_log = logging.getLogger(__name__)


def hdf5_to_pandas(
    hdf5_path: Union[str,List], # handle multiple files? They're not really handled in the checks (see dataset.py). 
    # should we make mandatory to combine them in one file? Here I consider only one hdf5 file, no lists.
    node_features: Union[List[str], str] = "all",
    edge_features: Union[List[str], str] = "all",
    target_features: Union[List[str], str] = "all"
):
    """Description.
    
    Args
    ----------
    hdf5_path (str or list): Path to hdf5 file(s). For multiple hdf5 files, 
        insert the paths in a list. Defaults to None.

    node_features (str or list, optional): consider all pre-computed node features ("all")
        or some defined node features (provide a list, example: ["res_type", "polarity", "bsa"]).
        The complete list can be found in deeprankcore/domain/nodestorage.py

    edge_features (list, optional): consider all pre-computed edge features ("all")
        or some defined edge features (provide a list, example: ["dist", "coulomb"]).
        The complete list can be found in deeprankcore/domain/edgestorage.py

    target_features (list, optional): consider all pre-computed target features ("all")
        or some defined target features (provide a list, example: ["binary", "capri_class"]).
        The complete list (only of the pre-defined ones) can be found in deeprankcore/domain/targetstorage.py
    Returns
    ----------        
    """
    # add target?
    with h5py.File(hdf5_path, 'r') as f:

        mol_key = list(f.keys())[0]
        
        # read available node features
        available_node_features = list(f[f"{mol_key}/{Nfeat.NODE}/"].keys())
        available_node_features = [key for key in available_node_features if key[0] != '_']  # ignore metafeatures
        
        # read available edge features
        available_edge_features = list(f[f"{mol_key}/{Efeat.EDGE}/"].keys())
        available_edge_features = [key for key in available_edge_features if key[0] != '_']  # ignore metafeatures

        # read available targets
        available_target_features = list(f[f"{mol_key}/{targets.VALUES}/"].keys())

        if node_features == "all":
            node_features = available_node_features
        if edge_features == "all":
            edge_features = available_edge_features
        if target_features == "all":
            target_features = available_target_features

        if not isinstance(node_features, list):
            node_features = [node_features]
        if not isinstance(edge_features, list):
            edge_features = [edge_features]
        if not isinstance(target_features, list):
            target_features = [target_features]

        # check node features
        for feat in node_features:
            if feat not in available_node_features:
                raise ValueError(
                    f"The node feature _{feat}_ was not found in the file {hdf5_path}.\
                    \nAvailable node features: {available_node_features}"
                )
        # check edge features
        for feat in edge_features:
            if feat not in available_edge_features:
                raise ValueError(
                    f"The edge feature _{feat}_ was not found in the file {hdf5_path}.\
                    \nAvailable edge features: {available_edge_features}"
                )
        # check target features
        for feat in target_features:
            if feat not in available_target_features:
                raise ValueError(
                    f"The target feature _{feat}_ was not found in the file {hdf5_path}.\
                    \nAvailable target features: {available_target_features}"
                )

        features = node_features + edge_features + target_features
        df_dict = {}
        for feat in features:
            df_dict['id'] = []
            df_dict[feat] = []
            for mol in f.keys():
                if feat in node_features:
                    df_dict[feat].append(f[mol]['node_features'][feat][:])
                elif feat in edge_features:
                    df_dict[feat].append(f[mol]['edge_features'][feat][:])
                else:
                    df_dict[feat].append(f[mol]['target_values'][feat][()])
                df_dict['id'].append(mol)
        df = pd.DataFrame(data=df_dict)
    return df