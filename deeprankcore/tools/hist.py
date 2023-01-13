import h5py
import pandas as pd
import logging

from typing import List, Union, Tuple
from deeprankcore.domain import (
    edgestorage as Efeat,
    nodestorage as Nfeat,
    targetstorage as targets)
import numpy as np
import matplotlib.pyplot as plt


_log = logging.getLogger(__name__)


def hdf5_to_pandas( # noqa: MC0001, pylint: disable=too-many-locals
    hdf5_path: Union[str,List],
    subset: List[str] = None,
    node_features: Union[List[str], str] = "all",
    edge_features: Union[List[str], str] = "all",
    target_features: Union[List[str], str] = "all"
) -> pd.DataFrame:
    """
    Args:
        hdf5_path (str or list): Path to hdf5 file(s). For multiple hdf5 files, 
            insert the paths in a list.

        subset (list, optional): list of keys from hdf5 file to include. Defaults to None (meaning include all).

        node_features (str or list, optional): consider all pre-computed node features ("all")
            or some defined node features (provide a list, example: ["res_type", "polarity", "bsa"]).
            The complete list can be found in deeprankcore/domain/nodestorage.py

        edge_features (list, optional): consider all pre-computed edge features ("all")
            or some defined edge features (provide a list, example: ["dist", "coulomb"]).
            The complete list can be found in deeprankcore/domain/edgestorage.py

        target_features (list, optional): consider all pre-computed target features ("all")
            or some defined target features (provide a list, example: ["binary", "capri_class"]).
            The complete list (only of the pre-defined ones) can be found in deeprankcore/domain/targetstorage.py
    
    Returns:
        df_final (pd.DataFrame): Pandas DataFrame containing the selected features as columns per all data points in
            hdf5_path files.   
    """
    if not isinstance(hdf5_path, list):
        hdf5_path = [hdf5_path]

    df_final = pd.DataFrame()

    for fname in hdf5_path:
        with h5py.File(fname, 'r') as f:

            mol_key = list(f.keys())[0]
            
            if subset is not None:
                mol_keys = [mol for mol, _ in f.items() if mol in subset]
            else:
                mol_keys = [mol for mol, _ in f.items()]
            
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

            df_dict = {}
            df_dict['id'] = mol_keys

            for feat in node_features + edge_features + target_features:
                if feat in node_features:
                    feat_type = 'node_features'
                elif feat in edge_features:
                    feat_type = 'edge_features'
                else:
                    feat_type = 'target_values'

                if f[mol_key][feat_type][feat][()].ndim == 2:
                    for i in range(f[mol_key][feat_type][feat][:].shape[1]):
                        df_dict[feat + '_' + str(i)] = [f[mol_key][feat_type][feat][:][:,i] for mol_key in mol_keys]
                else:
                    df_dict[feat] = [
                        f[mol_key][feat_type][feat][:]
                        if f[mol_key][feat_type][feat][()].ndim == 1
                        else f[mol_key][feat_type][feat][()] for mol_key in mol_keys]


            df = pd.DataFrame(data=df_dict)

        df_final = pd.concat([df_final, df])
    
    df_final.reset_index(drop=True, inplace=True)

    return df_final


def save_hist(
    df: pd.DataFrame,
    features: Union[str,List],
    fname: str,
    bins: Union[int,List,str] = 10,
    figsize: Tuple = (15, 15)
):
    """
    Args
    ----------
    df (pd.DataFrame): Pandas DataFrame object generated using hdf5_to_pandas function.

    features (str or list): features to be plotted. 

    fname (str): str or path-like or binary file-like object.

    bins (int or sequence or str): if bins is an integer, it defines the number of equal-width bins in the range.
        If bins is a sequence, it defines the bin edges, including the left edge of the first bin and the right edge
        of the last bin; in this case, bins may be unequally spaced. All but the last (righthand-most) bin is half-open.
        If bins is a string, it is one of the binning strategies supported by numpy.histogram_bin_edges:
        'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
        Defaults to 10.
    
    figsize (tuple): saved figure sizes, defaults to (15, 15).
    """
    if not isinstance(features, list):
        features = [features]

    means = [
        round(np.concatenate(df[feat].values).mean(), 1) if isinstance(df[feat].values[0], np.ndarray) \
        else round(df[feat].values.mean(), 1) \
        for feat in features]
    devs = [
        round(np.concatenate(df[feat].values).std(), 1) if isinstance(df[feat].values[0], np.ndarray) \
        else round(df[feat].values.std(), 1) \
        for feat in features]

    if len(features) > 1:

        fig, axs = plt.subplots(len(features), figsize=figsize)

        for row, feat in enumerate(features):

            if isinstance(df[feat].values[0], np.ndarray):
                axs[row].hist(np.concatenate(df[feat].values), bins=bins)
            else:
                axs[row].hist(df[feat].values, bins=bins)
            axs[row].set(xlabel=f'{feat} (mean {means[row]}, std {devs[row]})', ylabel='Count')
        fig.tight_layout()

    else:

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.hist(df[features[0]].values, bins=bins)
        ax.set(xlabel=f'{features[0]} (mean {means[0]}, std {devs[0]})', ylabel='Count')

    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
