import sys
import os
import re
import logging
import warnings
import numpy as np
import h5py
from typing import Callable, List, Union, Optional, Dict

from tqdm import tqdm
from ast import literal_eval
import torch
from torch_geometric.data.dataset import Dataset
from torch_geometric.data.data import Data

from deeprankcore.domain import (edgestorage as Efeat, nodestorage as Nfeat,
                                 targetstorage as targets, gridstorage)


_log = logging.getLogger(__name__)


class DeeprankDataset(Dataset):
    def __init__(self, # pylint: disable=too-many-arguments
                 hdf5_path: Union[str, List[str]],
                 subset: Union[List[str], None],
                 target: Union[str, None],
                 task: Union[str, None],
                 classes: Union[List[str], List[int], List[float], None],
                 use_tqdm: bool,
                 root_directory_path: str,
                 transform: Union[Callable, None],
                 pre_transform: Union[Callable, None],
                 target_filter: Union[Dict[str, str], None],
                 check_integrity: bool
    ):
        """
        Parent class of :class:`GridDataset` and :class:`GraphDataset` which inherits from :class:`torch_geometric.data.dataset.Dataset`.
            More detailed information about the parameters can be found in :class:`GridDataset` and :class:`GraphDataset`.
        """

        super().__init__(root_directory_path, transform, pre_transform)

        if isinstance(hdf5_path, str):
            self.hdf5_paths = [hdf5_path]

        elif isinstance(hdf5_path, list):
            self.hdf5_paths = hdf5_path

        else:
            raise TypeError(f"hdf5_path: unexpected type: {type(hdf5_path)}")

        self.use_tqdm = use_tqdm

        self.target = target
        self.subset = subset

        self.target_filter = target_filter
        
        if check_integrity:
            self._check_hdf5_files()

        self._check_task_and_classes(task, classes)

        # create the indexing system
        # alows to associate each mol to an index
        # and get fname and mol name from the index
        self._create_index_entries()

    def _check_hdf5_files(self):
        """Checks if the data contained in the .HDF5 file is valid."""
        _log.info("\nChecking dataset Integrity...")
        to_be_removed = []
        for hdf5_path in self.hdf5_paths:
            try:
                with h5py.File(hdf5_path, "r") as f5:
                    entry_names = list(f5.keys())
                    if len(entry_names) == 0:
                        _log.info(f"    -> {hdf5_path} is empty ")
                        to_be_removed.append(hdf5_path)
            except Exception as e:
                _log.error(e)
                _log.info(f"    -> {hdf5_path} is corrupted ")
                to_be_removed.append(hdf5_path)

        for hdf5_path in to_be_removed:
            self.hdf5_paths.remove(hdf5_path)

    def _check_task_and_classes(self, task: str, classes: Optional[str] = None):

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
                _log.info(f'Target classes set up to: {self.classes}')
            else:
                self.classes = classes

            self.classes_to_index = {
                class_: index for index, class_ in enumerate(self.classes)
            }
        else:
            self.classes = None
            self.classes_to_index = None

    def _create_index_entries(self):
        """Creates the indexing of each molecule in the dataset.

        Creates the indexing: [ ('1ak4.hdf5,1AK4_100w),...,('1fqj.hdf5,1FGJ_400w)]
        This allows to refer to one entry with its index in the list
        """
        _log.debug(f"Processing data set with .HDF5 files: {self.hdf5_paths}")

        self.index_entries = []

        desc = f"   {self.hdf5_paths}{' dataset':25s}"
        if self.use_tqdm:
            hdf5_path_iterator = tqdm(self.hdf5_paths, desc=desc, file=sys.stdout)
        else:
            _log.info(f"   {self.hdf5_paths} dataset\n")
            hdf5_path_iterator = self.hdf5_paths
        sys.stdout.flush()

        for hdf5_path in hdf5_path_iterator:
            if self.use_tqdm:
                hdf5_path_iterator.set_postfix(entry_name=os.path.basename(hdf5_path))
            try:
                with h5py.File(hdf5_path, "r") as hdf5_file:
                    if self.subset is None:
                        entry_names = list(hdf5_file.keys())
                    else:
                        entry_names = [entry_name for entry_name in self.subset if entry_name in list(hdf5_file.keys())]

                    for entry_name in entry_names:
                        if self._filter_targets(hdf5_file[entry_name]):
                            self.index_entries += [(hdf5_path, entry_name)]
            except Exception:
                _log.exception(f"on {hdf5_path}")

    def _filter_targets(self, entry_group: h5py.Group) -> bool:
        """
        Filters the entry according to a dictionary.
            The filter is based on the attribute self.target_filter that must be either
            of the form: { target_name : target_condition } or None.

        Args:
            entry_group(:class:`h5py.Group`): The entry group in the .HDF5 file.

        Returns:
            bool: True if we keep the entry False otherwise.

        Raises:
            ValueError: If an unsuported condition is provided.
        """

        if self.target_filter is None:
            return True

        for target_name, target_condition in self.target_filter.items():

            present_target_names = list(entry_group[targets.VALUES].keys())

            if target_name in present_target_names:

                # If we have a given target_condition, see if it's met.
                if isinstance(target_condition, str):

                    operation = target_condition
                    for operator_string in [">", "<", "==", "<=", ">=", "!="]:
                        operation = operation.replace(operator_string, "target_value" + operator_string)

                    if not literal_eval(operation):
                        return False

                elif target_condition is not None:
                    raise ValueError("Conditions not supported", target_condition)

            else:
                _log.warning(f"   :Filter {target_name} not found for entry {entry_group}\n"
                             f"   :Filter options are: {present_target_names}")
        return True

    def len(self) -> int:
        """
        Gets the length of the dataset, either :class:`GridDataset` or :class:`GraphDataset` object.

        Returns:
            int: Number of complexes in the dataset.
        """
        return len(self.index_entries)


# Grid features are stored per dimension and named accordingly.
# Example: position_001, position_002, position_003 (for x,y,z)
# Use this regular expression to take the feature name apart
GRID_PARTIAL_FEATURE_NAME_PATTERN = re.compile(r"^([a-zA-Z_]+)_([0-9]{3})$")


MAX_ENTRY_NAME_LENGTH = 100


class GridDataset(DeeprankDataset):
    def __init__( # pylint: disable=too-many-arguments
        self,
        hdf5_path: Union[str, list],
        subset: Optional[List[str]] = None,
        target: Optional[str] = None,
        task: Optional[str] = None,
        features: Optional[Union[List[str], str]] = "all",
        classes: Optional[Union[List[str], List[int], List[float]]] = None,
        tqdm: Optional[bool] = True,
        root: Optional[str] = "./",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        target_transform: Optional[bool] = False,
        target_filter: Optional[Dict[str, str]] = None,
        check_integrity: bool = True
    ):
        """
        Class to load the .HDF5 files data into grids.

        Args:
            hdf5_path (Union[str,list]): Path to .HDF5 file(s). For multiple .HDF5 files, insert the paths in a List. Defaults to None.

            subset (List[str], optional): List of keys from .HDF5 file to include. Defaults to None (meaning include all).

            target (str, optional): Default options are irmsd, lrmsd, fnat, bin, capri_class or dockq. It can also be a custom-defined target
                given to the Query class as input (see: `deeprankcore.query`); in this case, the task parameter needs to be explicitly specified as well.
                Only numerical target variables are supported, not categorical. If the latter is your case, please convert the categorical classes into
                numerical class indices before defining the :class:`GraphDataset` instance. Defaults to None.

            task (str, optional): 'regress' for regression or 'classif' for classification. Required if target not in
                ['irmsd', 'lrmsd', 'fnat', 'bin_class', 'capri_class', or 'dockq'], otherwise this setting is ignored.
                Automatically set to 'classif' if the target is 'bin_class' or 'capri_classes'.
                Automatically set to 'regress' if the target is 'irmsd', 'lrmsd', 'fnat' or 'dockq'.

            features (Union[List[str], str], optional): Consider all pre-computed features ("all") or some defined node features
                (provide a list, example: ["res_type", "polarity", "bsa"]). The complete list can be found in `deeprankcore.domain.gridstorage`. 

            classes (Union[List[str], List[int], List[float]], optional): Define the dataset target classes in classification mode. Defaults to [0, 1].

            tqdm (bool, optional): Show progress bar. Defaults to True.

            root (str, optional): Root directory where the dataset should be saved, defaults to "./".

            transform (Callable, optional): A function/transform that takes in a :class:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before every access. Defaults to None.

            pre_transform (Callable, optional):  A function/transform that takes in a :class:`torch_geometric.data.Data` object and returns
                a transformed version. The data object will be transformed before being saved to disk. Defaults to None.

            target_transform (bool, optional): Apply a log and then a sigmoid transformation to the target (for regression only).
                This puts the target value between 0 and 1, and can result in a more uniform target distribution and speed up the optimization.
                Defaults to False.
                
            target_filter (Dict[str, str], optional): Dictionary of type [target: cond] to filter the molecules.
                Note that the you can filter on a different target than the one selected as the dataset target. Defaults to None.

            check_integrity (bool, optional): Whether to check the integrity of the hdf5 files.
                Defaults to True.
        """
        super().__init__(hdf5_path, subset, target, task, classes, tqdm, root, transform, pre_transform, target_filter, check_integrity)

        self.features = features

        self._transform = transform
        self.target_transform = target_transform

        self._check_features()

    def _check_features(self):
        """Checks if the required features exist"""

        hdf5_path = self.hdf5_paths[0]

        # read available features
        with h5py.File(hdf5_path, "r") as hdf5_file:
            entry_name = list(hdf5_file.keys())[0]

            hdf5_all_feature_names = hdf5_file[f"{entry_name}/{gridstorage.MAPPED_FEATURES}"].keys()

            hdf5_matching_feature_names = []  # feature names that match with the requested list of names
            unpartial_feature_names = []  # feature names without their dimension number suffix

            for feature_name in hdf5_all_feature_names:

                if feature_name.startswith("_"):
                    continue  # ignore metafeatures

                partial_feature_match = GRID_PARTIAL_FEATURE_NAME_PATTERN.match(feature_name)
                if partial_feature_match is not None:  # there's a dimension number in the feature name

                    unpartial_feature_name = partial_feature_match.group(1)

                    if self.features == "all" or isinstance(self.features, list) and unpartial_feature_name in self.features:

                        hdf5_matching_feature_names.append(feature_name)

                    unpartial_feature_names.append(unpartial_feature_name)

                else:  # no numbers, it's a one-dimensional feature name

                    if self.features == "all" or isinstance(self.features, list) and feature_name in self.features:

                        hdf5_matching_feature_names.append(feature_name)

                    unpartial_feature_names.append(feature_name)

        # check for the requested features
        missing_features = []
        if self.features == "all":
            self.features = sorted(hdf5_all_feature_names)
        else:
            for feature_name in self.features:
                if feature_name not in unpartial_feature_names:
                    _log.info(f"The feature {feature_name} was not found in the file {hdf5_path}.")
                    missing_features.append(feature_name)

            self.features = sorted(hdf5_matching_feature_names)

        # raise error if any features are missing
        if len(missing_features) > 0:
            raise ValueError(
                f"Not all features could be found in the file {hdf5_path} under entry {entry_name}.\
                    \nMissing features are: {missing_features} \
                    \nCheck feature_modules passed to the preprocess function. \
                    \nProbably, the feature wasn't generated during the preprocessing step. \
                    Available features: {hdf5_all_feature_names}")

    def get(self, idx: int) -> Data:
        """Gets one grid item from its unique index.

        Args:
            idx(int): Index of the item, ranging from 0 to len(dataset).

        Returns:
            :class:`torch_geometric.data.data.Data`: item with tensors x, y if present, entry_names.
        """

        file_path, entry_name = self.index_entries[idx]
        return self.load_one_grid(file_path, entry_name)

    def load_one_grid(self, hdf5_path: str, entry_name: str) -> Data:
        """Loads one grid.

        Args:
            fname (str): .HDF5 file name.
            entry_name (str): Name of the entry.
            
        Returns:
            :class:`torch_geometric.data.data.Data`: item with tensors x, y if present, entry_names.
        """

        feature_data = []
        target_value = None

        with h5py.File(hdf5_path, 'r') as hdf5_file:
            entry_group = hdf5_file[entry_name]

            mapped_features_group = entry_group[gridstorage.MAPPED_FEATURES]
            for feature_name in self.features:
                feature_data.append(mapped_features_group[feature_name][gridstorage.FEATURE_VALUE][:])

            target_value = entry_group[targets.VALUES][self.target][()]

        # Wrap up the data in this object, for the collate_fn to handle it properly:
        data = Data(x=torch.tensor([feature_data], dtype=torch.float),
                    y=torch.tensor([target_value], dtype=torch.float))

        if len(entry_name) > MAX_ENTRY_NAME_LENGTH:
            raise ValueError(f"entry name '{entry_name}' is too long (max {MAX_ENTRY_NAME_LENGTH})")

        data.entry_names = torch.ByteTensor([bytes(entry_name.ljust(MAX_ENTRY_NAME_LENGTH), "utf8")])

        return data


class GraphDataset(DeeprankDataset):
    def __init__( # pylint: disable=too-many-arguments, too-many-locals
        self,
        hdf5_path: Union[str, List[str]],
        subset: Optional[List[str]] = None,
        target: Optional[str] = None,
        task: Optional[str] = None,
        node_features: Optional[Union[List[str], str]] = "all",
        edge_features: Optional[Union[List[str], str]] = "all",
        clustering_method: Optional[str] = None,
        classes: Optional[Union[List[str], List[int], List[float]]] = None,
        tqdm: Optional[bool] = True,
        root: Optional[str] = "./",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        edge_features_transform: Optional[Callable] = lambda x: np.tanh(-x / 2 + 2) + 1,
        target_transform: Optional[bool] = False,
        target_filter: Optional[Dict[str, str]] = None,
        check_integrity: bool = True
    ):
        """
        Class to load the .HDF5 files data into graphs.

        Args:
            hdf5_path (Union[str,list]): Path to .HDF5 file(s). For multiple .HDF5 files, insert the paths in a List. Defaults to None.

            subset (List[str], optional): List of keys from .HDF5 file to include. Defaults to None (meaning include all).

            target (str, optional): Default options are irmsd, lrmsd, fnat, bin, capri_class or dockq. It can also be a custom-defined target
                given to the Query class as input (see: `deeprankcore.query`); in this case, the task parameter needs to be explicitly specified as well.
                Only numerical target variables are supported, not categorical. If the latter is your case, please convert the categorical classes into
                numerical class indices before defining the :class:`GraphDataset` instance. Defaults to None.

            task (str, optional): 'regress' for regression or 'classif' for classification. Required if target not in
                ['irmsd', 'lrmsd', 'fnat', 'bin_class', 'capri_class', or 'dockq'], otherwise this setting is ignored.
                Automatically set to 'classif' if the target is 'bin_class' or 'capri_classes'.
                Automatically set to 'regress' if the target is 'irmsd', 'lrmsd', 'fnat' or 'dockq'.

            node_features (Union[List[str], str, optional): Consider all pre-computed node features ("all") or some defined node features
                (provide a list, example: ["res_type", "polarity", "bsa"]). The complete list can be found in `deeprankcore.domain.nodestorage`.

            edge_features (Union[List[str], str, optional): Consider all pre-computed edge features ("all") or some defined edge features
                (provide a list, example: ["dist", "coulomb"]). The complete list can be found in `deeprankcore.domain.edgestorage`.

            clustering_method (str, optional): "mcl" for Markov cluster algorithm (see https://micans.org/mcl/),
                or "louvain" for Louvain method (see https://en.wikipedia.org/wiki/Louvain_method).
                In both options, for each graph, the chosen method first finds communities (clusters) of nodes and generates
                a torch tensor whose elements represent the cluster to which the node belongs to. Each tensor is then saved in
                the .HDF5 file as a :class:`Dataset` called "depth_0". Then, all cluster members beloging to the same community are
                pooled into a single node, and the resulting tensor is used to find communities among the pooled clusters.
                The latter tensor is saved into the .HDF5 file as a :class:`Dataset` called "depth_1". Both "depth_0" and "depth_1"
                :class:`Datasets` belong to the "cluster" Group. They are saved in the .HDF5 file to make them available to networks
                that make use of clustering methods. Defaults to None.

            classes (Union[List[str], List[int], List[float]], optional): Define the dataset target classes in classification mode. Defaults to [0, 1].

            tqdm (bool, optional): Show progress bar. Defaults to True.

            root (str, optional): Root directory where the dataset should be saved, defaults to "./".

            transform (Callable, optional): A function/transform that takes in a :class:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before every access. Defaults to None.

            pre_transform (Callable, optional):  A function/transform that takes in a :class:`torch_geometric.data.Data` object and returns
                a transformed version. The data object will be transformed before being saved to disk. Defaults to None.

            edge_features_transform (Callable, optional): Transformation applied to the edge features. Defaults to lambda x: np.tanh(-x/2+2)+1.

            target_transform (bool, optional): Apply a log and then a sigmoid transformation to the target (for regression only).
                This puts the target value between 0 and 1, and can result in a more uniform target distribution and speed up the optimization.
                Defaults to False.

            target_filter (Dict[str, str], optional): Dictionary of type [target: cond] to filter the molecules.
                Note that the you can filter on a different target than the one selected as the dataset target. Defaults to None.

            check_integrity (bool, optional): Whether to check the integrity of the hdf5 files.
                Defaults to True.
        """
        super().__init__(hdf5_path, subset, target, task, classes, tqdm, root, transform, pre_transform, target_filter, check_integrity)

        self.node_features = node_features
        self.edge_features = edge_features
        self.clustering_method = clustering_method

        self._transform = transform
        self.edge_features_transform = edge_features_transform
        self.target_transform = target_transform

        self._check_features()

    def get(self, idx: int) -> Data:
        """
        Gets one graph item from its unique index.

        Args:
            idx(int): Index of the item, ranging from 0 to len(dataset).

        Returns:
            :class:`torch_geometric.data.data.Data`: item with tensors x, y if present, edge_index, edge_attr, pos, entry_names.
        """

        fname, mol = self.index_entries[idx]
        return self.load_one_graph(fname, mol)

    def load_one_graph(self, fname: str, entry_name: str)  -> Data: # pylint: disable = too-many-locals # noqa: MC0001
        """Loads one graph.

        Args:
            fname (str): .HDF5 file name.
            entry_name (str): Name of the entry.
            
        Returns:
            :class:`torch_geometric.data.data.Data`: item with tensors x, y if present, edge_index, edge_attr, pos, entry_names.
        """

        with h5py.File(fname, 'r') as f5:
            grp = f5[entry_name]

            # node features
            node_data = ()
            for feat in self.node_features:
                if feat[0] != '_':  # ignore metafeatures
                    vals = grp[f"{Nfeat.NODE}/{feat}"][()]
                    if vals.ndim == 1:
                        vals = vals.reshape(-1, 1)
                    node_data += (vals,)
            x = torch.tensor(np.hstack(node_data), dtype=torch.float)

            # edge index,
            # we have to have all the edges i.e : (i,j) and (j,i)
            if Efeat.INDEX in grp[Efeat.EDGE]:
                ind = grp[f"{Efeat.EDGE}/{Efeat.INDEX}"][()]
                if ind.ndim == 2:
                    ind = np.vstack((ind, np.flip(ind, 1))).T
                edge_index = torch.tensor(ind, dtype=torch.long).contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            # edge feature
            # we have to have all the edges i.e : (i,j) and (j,i)
            if (self.edge_features is not None 
                    and len(self.edge_features) > 0 
                    and Efeat.EDGE in grp):
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

            # target
            if self.target is None:
                y = None
            else:
                if targets.VALUES in grp and self.target in grp[targets.VALUES]:
                    y = torch.tensor([grp[f"{targets.VALUES}/{self.target}"][()]], dtype=torch.float).contiguous()

                    if self.task == targets.REGRESS and self.target_transform is True:
                        y = torch.sigmoid(torch.log(y))
                    elif self.task is not targets.REGRESS and self.target_transform is True:
                        raise ValueError(f"Task is set to {self.task}. Please set it to regress to transform the target with a sigmoid.")

                else:
                    possible_targets = grp[targets.VALUES].keys()
                    raise ValueError(f"Target {self.target} missing in entry {entry_name} in file {fname}, possible targets are {possible_targets}." +
                                     "\n Use the query class to add more target values to input data.")

            # positions
            pos = torch.tensor(grp[f"{Nfeat.NODE}/{Nfeat.POSITION}/"][()], dtype=torch.float).contiguous()

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
                                grp["clustering/" + self.clustering_method + "/depth_0"][()], dtype=torch.long)
                            cluster1 = torch.tensor(
                                grp["clustering/" + self.clustering_method + "/depth_1"][()], dtype=torch.long)
                        else:
                            _log.warning("no clusters detected")
                    else:
                        _log.warning(f"no clustering/{self.clustering_method} detected")

        # load
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)

        data.cluster0 = cluster0
        data.cluster1 = cluster1

        # entry name:
        if len(entry_name) > MAX_ENTRY_NAME_LENGTH:
            raise ValueError(f"entry name '{entry_name}' is too long (max {MAX_ENTRY_NAME_LENGTH})")

        data.entry_names = torch.ByteTensor([bytes(entry_name.ljust(MAX_ENTRY_NAME_LENGTH), "utf8")])

        # apply transformation
        if self._transform is not None:
            data = self._transform(data)

        return data

    def _check_features(self):
        """Checks if the required features exist"""
        f = h5py.File(self.hdf5_paths[0], "r")
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
                    _log.info(f"The node feature _{feat}_ was not found in the file {self.hdf5_paths[0]}.")
                    missing_node_features.append(feat)

        # check edge features
        missing_edge_features = []
        if self.edge_features == "all":
            self.edge_features = self.available_edge_features
        elif self.edge_features is not None:
            for feat in self.edge_features:
                if feat not in self.available_edge_features:
                    _log.info(f"The edge feature _{feat}_ was not found in the file {self.hdf5_paths[0]}.")
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
                f"Not all features could be found in the file {self.hdf5_paths[0]}.\
                    \nCheck feature_modules passed to the preprocess function. \
                    \nProbably, the feature wasn't generated during the preprocessing step. \
                    {miss_node_error}{miss_edge_error}")


def save_hdf5_keys(
    f_src_path: str,
    src_ids: List[str],
    f_dest_path: str,
    hardcopy = False
    ):
    """
    Save references to keys in src_ids in a new .HDF5 file.

    Args:
        f_src_path (str): The path to the .HDF5 file containing the keys.

        src_ids(List[str]): Keys to be saved in the new .HDF5 file. It should be a list containing at least one key.

        f_dest_path(str): The path to the new .HDF5 file.

        hardcopy(bool, optional): If False, the new file contains only references (external links, see :class:`ExternalLink` class from `h5py`)
            to the original .HDF5 file. If True, the new file contains a copy of the objects specified in src_ids (see h5py :class:`HardLink` from `h5py`).
            Default = False.
    """
    if not all(isinstance(d, str) for d in src_ids):
        raise TypeError("data_ids should be a list containing strings.")

    with h5py.File(f_dest_path,'w') as f_dest, h5py.File(f_src_path,'r') as f_src:
        for key in src_ids:
            if hardcopy:
                f_src.copy(f_src[key],f_dest)
            else:
                f_dest[key] = h5py.ExternalLink(f_src_path, "/" + key)
