import sys
import os
import logging
import warnings
import numpy as np
import h5py
from tqdm import tqdm
from ast import literal_eval
import torch
from torch_geometric.data.dataset import Dataset
from torch_geometric.data.data import Data
from typing import Callable, List, Union
from deeprankcore.domain import (edgestorage as Efeat, nodestorage as Nfeat,
                                targetstorage as targets)


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


class GraphDataset(Dataset):
    def __init__( # pylint: disable=too-many-arguments
        self,
        hdf5_path: Union[str,list],
        subset: list = None,
        target: str = None,
        node_features: Union[List[str], str] = "all",
        edge_features: Union[List[str], str] = "all",
        clustering_method: str = "mcl",
        task: str = None,
        classes: List = None,
        tqdm: bool = True,
        root: str = "./",
        transform: Callable = None,
        pre_transform: Callable = None,
        edge_features_transform: Callable = lambda x: np.tanh(-x / 2 + 2) + 1,
        target_filter: dict = None,
    ):
        """Class from which the hdf5 datasets are loaded.

        Args:
            hdf5_path (str or list): Path to hdf5 file(s). For multiple hdf5 files, 
                insert the paths in a list. Defaults to None.

            subset (list, optional): list of keys from hdf5 file to include. Defaults to None (meaning include all).

            target (str, optional): irmsd, lrmsd, fnat, bin, capri_class or dockq. It can also be a custom-defined
                target given to the Query class as input (see: deeprankcore.query); in the latter case, specify
                here its name. Only numerical target variables are supported, not categorical. If the latter is your case,
                please convert the categorical classes into numerical class indices before defining the GraphDataset instance.
                Defaults to None.

            node_features (str or list, optional): consider all pre-computed node features ("all")
                or some defined node features (provide a list, example: ["res_type", "polarity", "bsa"]).
                The complete list can be found in deeprankcore/domain/features.py

            edge_features (list, optional): consider all pre-computed edge features ("all")
                or some defined edge features (provide a list, example: ["dist", "coulomb"]).
                The complete list can be found in deeprankcore/domain/features.py

            clustering_method (str, optional): perform node clustering ('mcl', Markov Clustering,
                or 'louvain' algorithm). Note that this parameter can be None only if the neural
                network doesn't expects clusters (e.g. naive_gnn). Defaults to "mcl".

            task (str, optional): 'regress' for regression or 'classif' for classification.
                Used only if target not in ['irmsd', 'lrmsd', 'fnat', 'bin_class', 'capri_class', or 'dockq']
                Automatically set to 'classif' if the target is 'bin_class' or 'capri_classes'.
                Automatically set to 'regress' if the target is 'irmsd', 'lrmsd', 'fnat' or 'dockq'.

            classes (list, optional): define the dataset target classes in classification mode. Defaults to [0, 1].

            tqdm (bool, optional): Show progress bar. Defaults to True.

            root (str, optional): Root directory where the dataset should be
                saved. Defaults to "./"

            transform (callable, optional): A function/transform that takes in
                a torch_geometric.data.Data object and returns a transformed version.
                The data object will be transformed before every access. Defaults to None.

            pre_transform (callable, optional):  A function/transform that takes in
                a torch_geometric.data.Data object and returns a transformed version.
                The data object will be transformed before being saved to disk. Defaults to None.

            edge_features_transform (function, optional): transformation applied to the edge features.
                Defaults to lambdax:np.tanh(-x/2+2)+1.

            target_filter (dictionary, optional): Dictionary of type [target: cond] to filter the molecules.
                Note that the you can filter on a different target than the one selected as the dataset target.
                Defaults to None.
        """
        super().__init__(root, transform, pre_transform)

        if isinstance(hdf5_path, list):
            self.hdf5_path = hdf5_path
        else:
            self.hdf5_path = [hdf5_path]
        self.subset = subset
        self.target = target
        self.node_features = node_features
        self.edge_features = edge_features
        self.clustering_method = clustering_method
        self.tqdm = tqdm

        self._transform = transform
        self.edge_features_transform = edge_features_transform
        self.target_filter = target_filter

        self._check_hdf5_files()
        self._check_task_and_classes(task,classes)
        self._check_node_feature()
        self._check_edge_feature()

        # create the indexing system
        # alows to associate each mol to an index
        # and get fname and mol name from the index
        self._create_index_molecules()

        # get the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def len(self):
        """Gets the length of the dataset
        Returns:
            int: number of complexes in the dataset
        """
        return len(self.index_complexes)

    def get(self, index): # pylint: disable=arguments-renamed
        """Gets one item from its unique index.

        Args:
            index (int): index of the complex
        Returns:
            dict: {'mol':[fname,mol],'feature':feature,'target':target}
        """

        fname, mol = self.index_complexes[index]
        data = self.load_one_graph(fname, mol)
        return data

    def load_one_graph(self, fname, mol): # noqa
        """Loads one graph

        Args:
            fname (str): hdf5 file name
            mol (str): name of the molecule

        Returns:
            Data object or None: torch_geometric Data object containing the node features,
            the internal and external edge features, the target and the xyz coordinates.
            Return None if features cannot be loaded.
        """

        with h5py.File(fname, 'r') as f5:
            grp = f5[mol]

            # node features
            node_data = ()
            for feat in self.node_features:
                if feat[0] != '_':  # ignore metafeatures
                    vals = grp[f"{Nfeat.NODE}/{feat}"][()]
                    if vals.ndim == 1:
                        vals = vals.reshape(-1, 1)

                    node_data += (vals,)

            x = torch.tensor(np.hstack(node_data), dtype=torch.float).to(self.device)

            # edge index, we have to have all the edges i.e : (i,j) and (j,i)
            if Efeat.INDEX in grp[Efeat.EDGE]:
                ind = grp[f"{Efeat.EDGE}/{Efeat.INDEX}"][()]
                if ind.ndim == 2:
                    ind = np.vstack((ind, np.flip(ind, 1))).T
                edge_index = torch.tensor(ind, dtype=torch.long).contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_index = edge_index.to(self.device)

            # edge feature (same issue as above)
            if self.edge_features is not None and len(self.edge_features) > 0 and \
               Efeat.EDGE in grp:

                edge_data = ()
                for feat in self.edge_features:
                    if feat[0] != '_':   # ignore metafeatures
                        vals = grp[f"{Efeat.EDGE}/{feat}"][()]
                        if vals.ndim == 1:
                            vals = vals.reshape(-1, 1)
                        edge_data += (vals,)
                edge_data = np.hstack(edge_data)
                edge_data = np.vstack((edge_data, edge_data))
                edge_data = self.edge_features_transform(edge_data)
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
            if self.target is None:
                y = None
            else:
                if targets.VALUES in grp and self.target in grp[targets.VALUES]:
                    try:
                        y = torch.tensor([grp[f"{targets.VALUES}/{self.target}"][()]], dtype=torch.float).contiguous().to(self.device)
                    except Exception as e:
                        _log.error(e)
                        _log.info('If your target variable contains categorical classes, \
                        please convert them into class indices before defining the GraphDataset instance.')
                else:
                    possible_targets = grp[targets.VALUES].keys()
                    raise ValueError(f"Target {self.target} missing in entry {mol} in file {fname}, possible targets are {possible_targets}." +
                                     "\n Use the query class to add more target values to input data.")

            # positions
            pos = torch.tensor(grp[f"{Nfeat.NODE}/{Nfeat.POSITION}/"][()], dtype=torch.float).contiguous().to(self.device)

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

    def _check_hdf5_files(self):
        """Checks if the data contained in the hdf5 file is valid."""
        _log.info("\nChecking dataset Integrity...")
        remove_file = []
        for fname in self.hdf5_path:
            try:
                f = h5py.File(fname, "r")
                mol_names = list(f.keys())
                if len(mol_names) == 0:
                    _log.info(f"    -> {fname} is empty ")
                    remove_file.append(fname)
                f.close()
            except Exception as e:
                _log.error(e)
                _log.info(f"    -> {fname} is corrupted ")
                remove_file.append(fname)

        for name in remove_file:
            self.hdf5_path.remove(name)

    def _check_task_and_classes(self, task, classes):
        if self.target in [targets.IRMSD, targets.LRMSD, targets.FNAT, targets.DOCKQ]: 
            self.task = targets.REGRESS
        elif self.target in [targets.BINARY, targets.CAPRI]:
            self.task = targets.CLASSIF
        else:
            self.task = task
        
        if self.task not in [targets.CLASSIF, targets.REGRESS] and self.target is not None:
            raise ValueError(
                f"User target detected: {self.target} -> The task argument must be 'classif' or 'regress', currently set as {self.task}")
        if task != self.task and task is not None:
            warnings.warn(f"Target {self.target} expects {self.task}, but was set to task {task} by user.\n" +
                f"User set task is ignored and {self.task} will be used.")

        if self.task == targets.CLASSIF:
            if classes is None:
                self.classes = [0, 1]
            else:
                self.classes = classes

            self.classes_to_idx = {
                i: idx for idx, i in enumerate(self.classes)
            }
        else:
            self.classes = None
            self.classes_to_idx = None

    def _check_features(self):
        """Checks if the required features exist"""
        f = h5py.File(self.hdf5_path[0], "r")
        mol_key = list(f.keys())[0]
        
        # read available node features
        self.available_node_features = list(f[f"{mol_key}/{Nfeat.NODE}/"].keys())
        self.available_node_features = [key for key in self.available_node_features if key[0] != '_']  # ignore metafeatures
        
        # read available edge features
        self.available_edge_features = list(f[f"{mol_key}/{Efeat.EDGE}/"].keys())
        self.available_edge_features = [key for key in self.available_edge_features if key[0] != '_']  # ignore metafeatures

        f.close()

        # check node features
        missing_node_features = []
        if self.node_features == "all":
            self.node_features = self.available_node_features
        else:
            for feat in self.node_features:
                if feat not in self.available_node_features:
                    _log.info(f"The node feature _{feat}_ was not found in the file {self.hdf5_path[0]}.")
                    missing_node_features.append(feat)

        # check edge features
        missing_edge_features = []
        if self.edge_features == "all":
            self.edge_features = self.available_edge_features
        elif self.edge_features is not None:
            for feat in self.edge_features:
                if feat not in self.available_edge_features:
                    _log.info(f"The edge feature _{feat}_ was not found in the file {self.hdf5_path[0]}.")
                    missing_edge_features.append(feat)

        # raise error if any features are missing
        if missing_node_features + missing_edge_features:
            miss_node_error, miss_edge_error = "", ""
            _log.info("\nCheck feature_modules passed to the preprocess function.\
                Probably, the feature wasn't generated during the preprocessing step.")
            if missing_node_features:
                _log.info(f"\nAvailable node features: {self.available_node_features}\n")
                miss_node_error = f"\nMissing node features: {missing_node_features} \
                                    \nAvailable node features: {self.available_node_features}"
            if missing_edge_features:
                _log.info(f"\nAvailable edge features: {self.available_edge_features}\n")
                miss_edge_error = f"\nMissing edge features: {missing_edge_features} \
                                    \nAvailable edge features: {self.available_edge_features}"

            raise ValueError(
                f"Not all features could be found in the file {self.hdf5_path[0]}.\
                    \nCheck feature_modules passed to the preprocess function. \
                    \nProbably, the feature wasn't generated during the preprocessing step. \
                    {miss_node_error}{miss_edge_error}")

    def _create_index_molecules(self):
        """Creates the indexing of each molecule in the dataset.

        Creates the indexing: [ ('1ak4.hdf5,1AK4_100w),...,('1fqj.hdf5,1FGJ_400w)]
        This allows to refer to one complex with its index in the list
        """
        _log.debug(f"Processing data set with hdf5 files: {self.hdf5_path}")

        self.index_complexes = []

        desc = f"   {self.hdf5_path}{' dataset':25s}"
        if self.tqdm:
            data_tqdm = tqdm(self.hdf5_path, desc=desc, file=sys.stdout)
        else:
            _log.info(f"   {self.hdf5_path} dataset\n")
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
                    if self._filter(fh5[k]):
                        self.index_complexes += [(fdata, k)]
                fh5.close()
            except Exception:
                _log.exception(f"on {fdata}")

    def _filter(self, molgrp):
        """Filters the molecule according to a dictionary.

        The filter is based on the attribute self.target_filter
        that must be either of the form: { 'name' : cond } or None

        Args:
            molgrp (str): group name of the molecule in the hdf5 file
        Returns:
            bool: True if we keep the complex False otherwise
        Raises:
            ValueError: If an unsuported condition is provided
        """
        if self.target_filter is None:
            return True

        for cond_name, cond_vals in self.target_filter.items():

            try:
                molgrp[targets.VALUES][cond_name][()]
            except KeyError:
                _log.info(f"   :Filter {cond_name} not found for mol {molgrp}")
                _log.info("   :Filter options are")
                for k in molgrp[targets.VALUES].keys():
                    _log.info("   : ", k) # pylint: disable=logging-too-many-args

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
