import h5py
import pandas as pd
import logging
from typing import List, Union
from deeprankcore.domain import (
    edgestorage as Efeat,
    nodestorage as Nfeat,
    targetstorage as targets)


_log = logging.getLogger(__name__)

# in case savinbg the pd df into feather works (on Snellius, my home, the script read_hdf5_to_pandas.py),
# remember to add here a function and add pyarrow to the package's dependencies

def hdf5_to_pandas(
    hdf5_path: Union[str,List],
    node_features: Union[List[str], str] = "all",
    edge_features: Union[List[str], str] = "all",
    target_features: Union[List[str], str] = "all"
):
    """Description.
    
    Args
    ----------
    hdf5_path (str or list): Path to hdf5 file(s). For multiple hdf5 files, 
        insert the paths in a list.

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
    if isinstance(hdf5_path, list):
        hdf5_path = hdf5_path
    else:
        hdf5_path = [hdf5_path]

    df_final = pd.DataFrame()

    for fname in hdf5_path:
        with h5py.File(fname, 'r') as f:

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
            df_dict['id'] = [mol for mol in f.keys()]

            for feat in features:
                if feat in node_features:
                    feat_type = 'node_features'
                elif feat in edge_features:
                    feat_type = 'edge_features'
                else:
                    feat_type = 'target_values'

                dim = f[mol_key][feat_type][feat][()].ndim
                if dim == 2:
                    for i in range(f[mol_key][feat_type][feat][:].shape[1]):
                        df_dict[feat + '_' + str(i)] = [f[mol][feat_type][feat][:][:,i] for mol in f.keys()]
                else:
                    df_dict[feat] = [f[mol][feat_type][feat][:] if dim == 1 else f[mol][feat_type][feat][()] for mol in f.keys()]


            df = pd.DataFrame(data=df_dict)

        df_final = pd.concat([df_final, df])
    
    df_final.reset_index(drop=True, inplace=True)

    return df_final
