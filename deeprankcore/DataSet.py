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
from ast import literal_eval
from deeprankcore.domain.features import groups
from deeprankcore.domain import targettypes as targets
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


class HDF5DataSet(Dataset):
    def __init__( # pylint: disable=too-many-arguments
        self,
        hdf5_path,
        root: str = "./",
        transform: Callable = None,
        pre_transform: Callable = None,
        dict_filter: dict = None,
        target: str = None,
        task: str = None,
        classes: List = None,
        tqdm: bool = True,
        subset: list = None,
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

            target (str, optional): irmsd, lrmsd, fnat, bin, capri_class or dockq. It can also be a custom-defined
            target given to the Query class as input (see: deeprankcore.models.query); in the latter case, specify
            here its name. Only numerical target variables are supported, not categorical. If the latter is your case,
            please convert the categorical classes into numerical class indices before defining the HDF5DataSet instance.
            Defaults to None.

            task (str, optional): 'regress' for regression or 'classif' for classification.
                Used only if target not in ['irmsd', 'lrmsd', 'fnat', 'bin_class', 'capri_class', or 'dockq']
                Automatically set to 'classif' if the target is 'bin_class' or 'capri_classes'.
                Automatically set to 'regress' if the target is 'irmsd', 'lrmsd', 'fnat' or 'dockq'.

            classes (list, optional): define the dataset target classes in classification mode. Defaults to [0, 1].

            tqdm (bool, optional): Show progress bar. Defaults to True.

            subset (list, optional): list of keys from hdf5 file to include. Defaults to None (meaning include all).

            node_feature (str or list, optional): consider all pre-computed node features ("all")
            or some defined node features (provide a list, example: ["res_type", "polarity", "bsa"]).
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

        self.target = target
        if self.target in [targets.IRMSD, targets.LRMSD, targets.FNAT, targets.DOCKQ]: 
            self.task = targets.REGRESS
        elif self.target in [targets.BINARY, targets.CAPRI]:
            self.task = targets.CLASSIF
        else:
            self.task = task
        
        if self.task not in [targets.CLASSIF, targets.REGRESS] and self.target is not None:
            raise ValueError(
                f"User target detected: {self.target} -> The task argument must be 'classif' or 'regress', currently set as {self.task} \n\t"
                "Example: \n\t"
                ""
                "model = NeuralNet(dataset, GINet,"
                "                  target='physiological_assembly',"
                "                  task='classif')")
        
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

        self.dict_filter = dict_filter
        self.tqdm = tqdm
        self.subset = subset

        self.node_feature = node_feature

        self.edge_feature = edge_feature

        self.edge_feature_transform = edge_feature_transform
        self._transform = transform

        self.clustering_method = clustering_method

        # check if the files are ok
        self._check_hdf5_files()

        # check the selection of features
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

    def _check_node_feature(self):
        """Checks if the required node features exist"""
        f = h5py.File(self.hdf5_path[0], "r")
        mol_key = list(f.keys())[0]
        self.available_node_feature = list(f[f"{mol_key}/{groups.NODE}/"].keys())
        self.available_node_feature = [key for key in self.available_node_feature if key[0] != '_'] # ignore metafeatures
        f.close()

        if self.node_feature == "all":
            self.node_feature = self.available_node_feature
        else:
            for feat in self.node_feature:
                if feat not in self.available_node_feature:
                    _log.info(f"The node feature _{feat}_ was not found in the file {self.hdf5_path[0]}.")
                    _log.info("\nCheck feature_modules passed to the preprocess function.\
                        Probably, the feature wasn't generated during the preprocessing step.")
                    _log.info(f"\nPossible node features: {self.available_node_feature}\n")
                    sys.exit()

    def _check_edge_feature(self):
        """Checks if the required edge features exist"""
        f = h5py.File(self.hdf5_path[0], "r")
        mol_key = list(f.keys())[0]
        self.available_edge_feature = list(f[f"{mol_key}/{groups.EDGE}/"].keys())
        self.available_edge_feature = [key for key in self.available_edge_feature if key[0] != '_'] # ignore metafeatures
        f.close()

        if self.edge_feature == "all":
            self.edge_feature = self.available_edge_feature
        elif self.edge_feature is not None:
            for feat in self.edge_feature:
                if feat not in self.available_edge_feature:
                    _log.info(f"The edge feature _{feat}_ was not found in the file {self.hdf5_path[0]}.")
                    _log.info("\nCheck feature_modules passed to the preprocess function.\
                        Probably, the feature wasn't generated during the preprocessing step.")
                    _log.info(f"\nPossible edge features: {self.available_edge_feature}\n")
                    sys.exit()

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
            for feat in self.node_feature:
                if feat[0] != '_':  # ignore metafeatures
                    vals = grp[f"{groups.NODE}/{feat}"][()]
                    if vals.ndim == 1:
                        vals = vals.reshape(-1, 1)

                    node_data += (vals,)

            x = torch.tensor(np.hstack(node_data), dtype=torch.float).to(self.device)

            # edge index, we have to have all the edges i.e : (i,j) and (j,i)
            if groups.INDEX in grp[groups.EDGE]:
                ind = grp[f"{groups.EDGE}/{groups.INDEX}"][()]
                if ind.ndim == 2:
                    ind = np.vstack((ind, np.flip(ind, 1))).T
                edge_index = torch.tensor(ind, dtype=torch.long).contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_index = edge_index.to(self.device)

            # edge feature (same issue as above)
            if self.edge_feature is not None and len(self.edge_feature) > 0 and \
               groups.EDGE in grp:

                edge_data = ()
                for feat in self.edge_feature:
                    if feat[0] != '_':   # ignore metafeatures
                        vals = grp[f"{groups.EDGE}/{feat}"][()]
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
            if self.target is None:
                y = None
            else:
                if targets.VALUES in grp and self.target in grp[targets.VALUES]:
                    try:
                        y = torch.tensor([grp[f"{targets.VALUES}/{self.target}"][()]], dtype=torch.float).contiguous().to(self.device)
                    except Exception as e:
                        _log.error(e)
                        _log.info('If your target variable contains categorical classes, \
                        please convert them into class indices before defining the HDF5DataSet instance.')
                else:
                    possible_targets = grp[targets.VALUES].keys()
                    raise ValueError(f"Target {self.target} missing in entry {mol} in file {fname}, possible targets are {possible_targets}." +
                                     "\n Use the query class to add more target values to input data.")

            # positions
            pos = torch.tensor(grp[f"{groups.NODE}/{groups.POSITION}/"][()], dtype=torch.float).contiguous().to(self.device)

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
