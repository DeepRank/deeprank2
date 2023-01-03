import h5py
import pandas as pd
import logging
from typing import List, Union
from deeprankcore.domain import (
    edgestorage as Efeat,
    nodestorage as Nfeat,
    targetstorage as targets)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


_log = logging.getLogger(__name__)


def hdf5_to_pandas( # noqa: MC0001, pylint: disable=too-many-locals
    hdf5_path: Union[str,List],
    subset: List[str] = None,
    node_features: Union[List[str], str] = "all",
    edge_features: Union[List[str], str] = "all",
    target_features: Union[List[str], str] = "all"
) -> pd.DataFrame:
    """
    Args
    ----------
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
    Returns
    ----------
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


def plot_distr(
    df: pd.DataFrame,
    features: Union[str,List],
) -> go.Figure():
    """
    Args
    ----------
    df (pd.DataFrame): Pandas DataFrame object generated using hdf5_to_pandas function.

    features (str or list): features to be plotted. 

    Returns
    ----------
    fig (pd.DataFrame): go.Figure() object containing the distributions of the chosen features.   
    """
    if not isinstance(features, list):
        features = [features]

    means = [round(df[feat].apply(lambda x: x.mean() if isinstance(x, np.ndarray) else x).mean(), 1) for feat in features]
    devs = [
        round(df[feat].apply(lambda x: x.std() if isinstance(x, np.ndarray) else x).mean(), 1)
        if isinstance(df[feat].loc[0], np.ndarray)
        else round(df[feat].std(), 1) for feat in features]

    fig = make_subplots(
        rows=len(features),
        cols=1,
        subplot_titles=[
            f'{features[idx]} (mean {means[idx]}, std {devs[idx]})' \
            for idx in range(len(features))
        ])

    for row, feature in enumerate(features):
        if isinstance(df[feature].loc[0], np.ndarray):
            for idx in range(df.shape[0]):
                fig.add_trace(
                    go.Histogram(x=df[feature][idx]),
                    row=row+1,
                    col=1)
        else:
            fig.add_trace(
                go.Histogram(x=df[feature]),
                row=row+1, col=1)

    fig.update_layout(
        barmode='stack',
        title='Chosen features',
        showlegend = False,
        xaxis_title='Value',
        yaxis_title='Count')
    fig.update_traces(opacity=0.75)

    return fig
