import sys
import os

import logging
import warnings

import torch
import numpy as np
from torch_geometric.data.dataset import Dataset
from torch_geometric.data.data import Data
from tqdm import tqdm
import h5py
import copy
from ast import literal_eval
from deeprankcore.community_pooling import community_detection, community_pooling
from typing import Callable, List, Union


_log = logging.getLogger(__name__)

def save_hdf5_keys(
    f_src_path: str,
    src_ids: List[str],
    f_dest_path: str,
    hardcopy = False
    ):
    """Save references to keys in data_ids in a new hdf5 file.
    Parameters
    ----------
    f_src_path : str
        The path to the hdf5 file containing the keys.
    src_ids : List[str]
        Keys to be saved in the new hdf5 file.
        It should be a list containing at least one key.
    f_dest_path : str
        The path to the new hdf5 file.
    hardcopy : bool, default = False
        If False, the new file contains only references.
        (external links, see h5py ExternalLink class) to the original hdf5 file.
        If True, the new file contains a copy of the objects specified in data_ids
        (see h5py HardLink class).
        
    """
    if not all(isinstance(d, str) for d in src_ids):
        raise TypeError("data_ids should be a list containing strings.")

    with h5py.File(f_dest_path,'w') as f_dest, h5py.File(f_src_path,'r') as f_src:
        for key in src_ids:
            if hardcopy:
                f_src.copy(f_src[key],f_dest)
            else:
                f_dest[key] = h5py.ExternalLink(f_src_path, "/" + key)


def _DivideDataSet(dataset, val_size=None):
    """Divides the dataset into a training set and an evaluation set

    Args:
        dataset (HDF5DataSet): input dataset to be split into training and validation data
        val_size (float or int, optional): fraction of dataset (if float) or number of datapoints (if int) to use for validation. 
            Defaults to 0.25.

    Returns:
        HDF5DataSet: [description]
    """

    if val_size is None:
        val_size = 0.25
    full_size = len(dataset)

    # find number of datapoints to include in training dataset
    if isinstance (val_size, float):
        n_val = int(val_size * full_size)
    elif isinstance (val_size, int):
        n_val = val_size
    else:
        raise TypeError (f"type(val_size) must be float, int or None ({type(val_size)} detected.)")
    
    # raise exception if no training data or negative validation size
    if n_val >= full_size or n_val < 0:
        raise ValueError ("invalid val_size. \n\t" +
            f"val_size must be a float between 0 and 1 OR an int smaller than the size of the dataset used ({full_size})")

    index = np.arange(full_size)
    np.random.shuffle(index)
    index_train, index_val = index[n_val:], index[:n_val]

    dataset_train = copy.deepcopy(dataset)
    dataset_train.index_complexes = [dataset.index_complexes[i] for i in index_train]

    dataset_val = copy.deepcopy(dataset)
    dataset_val.index_complexes = [dataset.index_complexes[i] for i in index_val]

    return dataset_train, dataset_val


def PreCluster(dataset, method):
    """Pre-clusters nodes of the graphs

    Args:
        dataset (HDF5DataSet object)
        method (srt): 'mcl' (Markov Clustering) or 'louvain'
    """
    for fname, mol in tqdm(dataset.index_complexes):

        data = dataset.load_one_graph(fname, mol)

        if data is None:
            f5 = h5py.File(fname, "a")
            try:
                print(f"deleting {mol}")
                del f5[mol]
            except BaseException:
                print(f"{mol} not found")
            f5.close()
            continue

        f5 = h5py.File(fname, "a")
        grp = f5[mol]

        clust_grp = grp.require_group("clustering")

        if method.lower() in clust_grp:
            print(f"Deleting previous data for mol {mol} method {method}")
            del clust_grp[method.lower()]

        method_grp = clust_grp.create_group(method.lower())

        cluster = community_detection(
            data.edge_index, data.num_nodes, method=method
        )
        method_grp.create_dataset("depth_0", data=cluster.cpu())

        data = community_pooling(cluster, data)

        cluster = community_detection(
            data.edge_index, data.num_nodes, method=method
        )
        method_grp.create_dataset("depth_1", data=cluster.cpu())

        f5.close()


class HDF5DataSet(Dataset):
    def __init__( # pylint: disable=too-many-arguments
        self,
        hdf5_path: str,
        root: str = "./",
        transform: Callable = None,
        pre_transform: Callable = None,
        dict_filter: dict = None,
        tqdm: bool = True,
        subset: List[str] = None,
        node_feature: Union[List[str], str] = "all",
        edge_feature: Union[List[str], str] = "all",
        clustering_method: str = "mcl",
        edge_feature_transform: Callable = lambda x: np.tanh(-x / 2 + 2) + 1,
    ):
        """Class from which the hdf5 datasets are loaded.

        Args:
            root (str, optional): Root directory where the dataset should be
            saved. Defaults to "./"

            hdf5_path (str, optional): Path to hdf5 file(s). For multiple hdf5 files, 
            insert the paths in a list. Defaults to None.

            transform (callable, optional): A function/transform that takes in
            a torch_geometric.data.Data object and returns a transformed version.
            The data object will be transformed before every access. Defaults to None.

            pre_transform (callable, optional):  A function/transform that takes in
            a torch_geometric.data.Data object and returns a transformed version.
            The data object will be transformed before being saved to disk. Defaults to None.

            dict_filter (dictionary, optional): Dictionary of type [name: cond] to filter the molecules.
            Defaults to None.

            tqdm (bool, optional): Show progress bar. Defaults to True.

            subset (list, optional): list of keys from hdf5 file to include. Defaults to None (meaning include all).

            node_feature (str or list, optional): consider all pre-computed node features ("all")
            or some defined node features (provide a list, example: ["type", "polarity", "bsa"]).
            The complete list can be found in deeprankcore/domain/features.py

            edge_feature (list, optional): consider all pre-computed edge features ("all")
            or some defined edge features (provide a list, example: ["dist", "coulomb"]).
            The complete list can be found in deeprankcore/domain/features.py

            clustering_method (str, optional): perform node clustering ('mcl', Markov Clustering,
            or 'louvain' algorithm). Note that this parameter can be None only if the neural
            network doesn't expects clusters (e.g. naive_gnn). Defaults to "mcl".

            edge_feature_transform (function, optional): transformation applied to the edge features.
            Defaults to lambdax:np.tanh(-x/2+2)+1.
        """
        super().__init__(root, transform, pre_transform)

        # allow for multiple hdf5 files
        self.hdf5_path = hdf5_path
        if not isinstance(hdf5_path, list):
            self.hdf5_path = [hdf5_path]

        self.dict_filter = dict_filter
        self.tqdm = tqdm
        self.subset = subset

        self.node_feature = node_feature

        self.edge_feature = edge_feature

        self.edge_feature_transform = edge_feature_transform
        self._transform = transform

        self.clustering_method = clustering_method

        # check if the files are ok
        self.check_hdf5_files()

        # check the selection of features
        self.check_node_feature()
        self.check_edge_feature()

        # create the indexing system
        # alows to associate each mol to an index
        # and get fname and mol name from the index
        self.create_index_molecules()

        # get the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def len(self):
        """Gets the length of the dataset
        Returns:
            int: number of complexes in the dataset
        """
        return len(self.index_complexes)

    def _download(self):
        pass

    def _process(self):
        pass

    def get(self, index): # pylint: disable=arguments-renamed
        """Gets one item from its unique index.

        Args:
            index (int): index of the complex
        Returns:
            dict: {'mol':[fname,mol],'feature':feature,'target':target}
        """
        # I NEED TO CHECK WHAT TARGET DOES HERE, BUT PROBS OK

        fname, mol = self.index_complexes[index]
        data = self.load_one_graph(fname, mol)
        return data

    def check_hdf5_files(self):
        """Checks if the data contained in the hdf5 file is valid."""
        print("\nChecking dataset Integrity...\n")
        remove_file = []
        for fname in self.hdf5_path:
            try:
                f = h5py.File(fname, "r")
                mol_names = list(f.keys())
                if len(mol_names) == 0:
                    print(f"    -> {fname} is empty ")
                    remove_file.append(fname)
                f.close()
            except Exception as e:
                print(e)
                print(f"    -> {fname} is corrupted ")
                remove_file.append(fname)

        for name in remove_file:
            self.hdf5_path.remove(name)

    def check_node_feature(self):
        """Checks if the required node features exist"""
        f = h5py.File(self.hdf5_path[0], "r")
        mol_key = list(f.keys())[0]
        self.available_node_feature = list(f[mol_key + "/node_data/"].keys())
        f.close()

        if self.node_feature == "all":
            self.node_feature = self.available_node_feature
        else:
            for feat in self.node_feature:
                if feat not in self.available_node_feature:
                    print(f"The node feature _{feat}_ was not found in the file {self.hdf5_path[0]}.")
                    print("\nCheck feature_modules passed to the preprocess function. Probably, the feature wasn't generated during the preprocessing step.")
                    print(f"\nPossible node features: {self.available_node_feature}\n")
                    sys.exit()

    def check_edge_feature(self):
        """Checks if the required edge features exist"""
        f = h5py.File(self.hdf5_path[0], "r")
        mol_key = list(f.keys())[0]
        self.available_edge_feature = list(f[mol_key + "/edge_data/"].keys())
        f.close()

        if self.edge_feature == "all":
            self.edge_feature = self.available_edge_feature
        elif self.edge_feature is not None:
            for feat in self.edge_feature:
                if feat not in self.available_edge_feature:
                    print(f"The edge feature _{feat}_ was not found in the file {self.hdf5_path[0]}.")
                    print("\nCheck feature_modules passed to the preprocess function. Probably, the feature wasn't generated during the preprocessing step.")
                    print(f"\nPossible edge features: {self.available_edge_feature}\n")
                    sys.exit()

    def load_one_graph(self, fname, mol): # noqa
        """Loads one graph

        Args:
            fname (str): hdf5 file name
            mol (str): name of the molecule

        Returns:
            Data object or None: torch_geometric Data object containing the node features,
            the internal and external edge features, the target, and the xyz coordinates (position).
            Return None if features cannot be loaded.
        """

        with h5py.File(fname, 'r') as f5:
            grp = f5[mol]

            # node features
            node_data = ()
            for feat in self.node_feature:
                vals = grp["node_data/" + feat][()]
                if vals.ndim == 1:
                    vals = vals.reshape(-1, 1)

                node_data += (vals,)

            x = torch.tensor(np.hstack(node_data), dtype=torch.float).to(self.device)

            # edge index, we have to have all the edges i.e : (i,j) and (j,i)
            if "edge_index" in grp:
                ind = grp['edge_index'][()]
                if ind.ndim == 2:
                    ind = np.vstack((ind, np.flip(ind, 1))).T
                edge_index = torch.tensor(ind, dtype=torch.long).contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_index = edge_index.to(self.device)

            # edge feature (same issue as above)
            if self.edge_feature is not None and len(self.edge_feature) > 0 and \
               "edge_data" in grp:

                edge_data = ()
                for feat in self.edge_feature:
                    vals = grp['edge_data/'+feat][()]
                    if vals.ndim == 1:
                        vals = vals.reshape(-1, 1)
                    edge_data += (vals,)
                edge_data = np.hstack(edge_data)
                edge_data = np.vstack((edge_data, edge_data))
                edge_data = self.edge_feature_transform(edge_data)
                edge_attr = torch.tensor(edge_data, dtype=torch.float).contiguous()
            else:
                edge_attr = torch.empty((edge_index.shape[1], 0), dtype=torch.float).contiguous()
            edge_attr = edge_attr.to(self.device)

            if any(key in grp for key in ("internal_edge_index", "internal_edge_data")):
                warnings.warn(
                    """Internal edges are not supported anymore.
                    You should probably prepare the hdf5 file
                    with a more up to date version of this software.""", DeprecationWarning)

            # target
            # NOT SURE HOW TO DEAL WITH THIS IF TARGET IS ONLY SET IN NEURALNET.
            #   DATASET APPEARS TO INHERENTLY NEED A TARGET IN ORDER TO CREATE GRAPHS. 
            #   NOT SURE WHAT WOULD HAPPEN IF WE REMOVE THE TARGET FROM THE GRAPHS. 
            #   ALTERNATIVELY, IT DOES SEEM LIKE ALL POSSIBLE TARGETS ARE BEING READ HERE ANYWAY, MAYBE THIS DOES NOT HAVE TO BE SET BY USER
            if self.target is None:
                y = None
            else:
                if "score" in grp and self.target in grp["score"]:
                    y = torch.tensor([grp['score/'+self.target][()]], dtype=torch.float).contiguous().to(self.device)
                else:
                    possible_targets = grp["score"].keys()
                    raise ValueError(f"Target {self.target} missing in entry {mol} in file {fname}, possible targets are {possible_targets}." +
                                     " Use the query class to add more target values to input data.")

            # positions
            pos = torch.tensor(grp['node_data/pos/'][()], dtype=torch.float).contiguous().to(self.device)

            # cluster
            cluster0 = None
            cluster1 = None
            if self.clustering_method is not None:
                if 'clustering' in grp.keys():
                    if self.clustering_method in grp["clustering"].keys():
                        if (
                            "depth_0" in grp[f"clustering/{self.clustering_method}"].keys() and
                            "depth_1" in grp[f"clustering/{self.clustering_method}"].keys()
                            ):

                            cluster0 = torch.tensor(
                                grp["clustering/" + self.clustering_method + "/depth_0"][()], dtype=torch.long).to(self.device)
                            cluster1 = torch.tensor(
                                grp["clustering/" + self.clustering_method + "/depth_1"][()], dtype=torch.long).to(self.device)
                        else:
                            _log.warning("no clusters detected")
                    else:
                        _log.warning(f"no clustering/{self.clustering_method} detected")
                else:
                    _log.warning("no clustering group found")
            else:
                _log.warning("no cluster method set")

        # load
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)

        data.cluster0 = cluster0
        data.cluster1 = cluster1

        # mol name
        data.mol = mol

        # apply transformation
        if self._transform is not None:
            data = self._transform(data)

        return data

    def create_index_molecules(self):
        """Creates the indexing of each molecule in the dataset.

        Creates the indexing: [ ('1ak4.hdf5,1AK4_100w),...,('1fqj.hdf5,1FGJ_400w)]
        This allows to refer to one complex with its index in the list
        """
        _log.debug(f"Processing data set with hdf5 files: {self.hdf5_path}")

        self.index_complexes = []

        desc = f"{'   Train dataset':25s}"
        if self.tqdm:
            data_tqdm = tqdm(self.hdf5_path, desc=desc, file=sys.stdout)
        else:
            print("   Train dataset")
            data_tqdm = self.hdf5_path
        sys.stdout.flush()

        for fdata in data_tqdm:
            if self.tqdm:
                data_tqdm.set_postfix(mol=os.path.basename(fdata))
            try:
                fh5 = h5py.File(fdata, "r")
                if self.subset is None:
                    mol_names = list(fh5.keys())
                else:
                    mol_names = [i for i in self.subset if i in list(fh5.keys())]

                for k in mol_names:
                    if self.filter(fh5[k]):
                        self.index_complexes += [(fdata, k)]
                fh5.close()
            except Exception:
                _log.exception(f"on {fdata}")

    def filter(self, molgrp):
        """Filters the molecule according to a dictionary.

        The filter is based on the attribute self.dict_filter
        that must be either of the form: { 'name' : cond } or None

        Args:
            molgrp (str): group name of the molecule in the hdf5 file
        Returns:
            bool: True if we keep the complex False otherwise
        Raises:
            ValueError: If an unsuported condition is provided
        """
        if self.dict_filter is None:
            return True

        for cond_name, cond_vals in self.dict_filter.items():

            try:
                molgrp["score"][cond_name][()]
            except KeyError:
                print(f"   :Filter {cond_name} not found for mol {molgrp}")
                print("   :Filter options are")
                for k in molgrp["score"].keys():
                    print("   : ", k)

            # if we have a string it's more complicated
            if isinstance(cond_vals, str):

                ops = [">", "<", "=="]
                new_cond_vals = cond_vals
                for o in ops:
                    new_cond_vals = new_cond_vals.replace(o, "val" + o)

                if not literal_eval(new_cond_vals):
                    return False
            else:
                raise ValueError("Conditions not supported", cond_vals)

        return True
