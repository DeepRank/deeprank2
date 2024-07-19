from __future__ import annotations

import inspect
import logging
import os
import pickle
import re
import sys
import warnings
from typing import Literal

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.data.data import Data
from torch_geometric.data.dataset import Dataset
from tqdm import tqdm

from deeprank2.domain import edgestorage as Efeat
from deeprank2.domain import gridstorage
from deeprank2.domain import nodestorage as Nfeat
from deeprank2.domain import targetstorage as targets

_log = logging.getLogger(__name__)


class DeeprankDataset(Dataset):
    """Parent class of :class:`GridDataset` and :class:`GraphDataset`.

    This class inherits from :class:`torch_geometric.data.dataset.Dataset`.
    More detailed information about the parameters can be found in :class:`GridDataset` and :class:`GraphDataset`.
    """

    def __init__(
        self,
        hdf5_path: str | list[str],
        subset: list[str] | None,
        train_source: str | GridDataset | GraphDataset | None,
        target: str | None,
        target_transform: bool | None,
        target_filter: dict[str, str] | None,
        task: str | None,
        classes: list[str] | list[int] | list[float] | None,
        use_tqdm: bool,
        root: str,
        check_integrity: bool,
    ):
        super().__init__(root)

        if isinstance(hdf5_path, str):
            self.hdf5_paths = [hdf5_path]

        elif isinstance(hdf5_path, list):
            self.hdf5_paths = hdf5_path

        else:
            msg = f"hdf5_path: unexpected type: {type(hdf5_path)}"
            raise TypeError(msg)

        self.subset = subset
        self.train_source = train_source
        self.target = target
        self.target_transform = target_transform
        self.target_filter = target_filter

        if check_integrity:
            self._check_hdf5_files()

        self._check_task_and_classes(task, classes)
        self.use_tqdm = use_tqdm

        # create the indexing system
        # alows to associate each mol to an index
        # and get fname and mol name from the index
        self._create_index_entries()

        self.df = None
        self.means = None
        self.devs = None
        self.train_means = None
        self.train_devs = None

        # get the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _check_and_inherit_train(  # noqa: C901
        self,
        data_type: GridDataset | GraphDataset,
        inherited_params: list[str],
    ) -> None:
        """Check if the pre-trained model or training set provided are valid for validation and/or testing, and inherit the parameters."""
        if isinstance(self.train_source, str):
            try:
                if torch.cuda.is_available():
                    data = torch.load(self.train_source)
                else:
                    data = torch.load(self.train_source, map_location=torch.device("cpu"))
                if data["data_type"] is not data_type:
                    msg = (
                        f"The pre-trained model has been trained with data of type {data['data_type']}, but you are trying \n\t"
                        f"to define a {data_type}-class validation/testing dataset. Please provide a valid DeepRank2 \n\t"
                        f"model trained with {data_type}-class type data, or define the dataset using the appropriate class."
                    )
                    raise TypeError(msg)
                if data_type is GraphDataset:
                    self.train_means = data["means"]
                    self.train_devs = data["devs"]
                    # convert strings in 'transform' key to lambda functions
                    if data["features_transform"]:
                        for key in data["features_transform"].values():
                            if key["transform"] is None:
                                continue
                            key["transform"] = eval(key["transform"])  # noqa: S307, PGH001
            except pickle.UnpicklingError as e:
                msg = "The path provided to `train_source` is not a valid DeepRank2 pre-trained model."
                raise ValueError(msg) from e
        elif isinstance(self.train_source, data_type):
            data = self.train_source
            if data_type is GraphDataset:
                self.train_means = self.train_source.means
                self.train_devs = self.train_source.devs
        else:
            msg = (
                f"The train data provided is invalid: {type(self.train_source)}.\n\t"
                f"Please provide a valid training {data_type} or the path to a valid DeepRank2 pre-trained model."
            )
            raise TypeError(msg)

        # match parameters with the ones in the training set
        self._check_inherited_params(inherited_params, data)

    def _check_hdf5_files(self) -> None:
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
            except Exception as e:  # noqa: BLE001, PERF203
                _log.error(e)
                _log.info(f"    -> {hdf5_path} is corrupted ")
                to_be_removed.append(hdf5_path)

        for hdf5_path in to_be_removed:
            self.hdf5_paths.remove(hdf5_path)

    def _check_task_and_classes(self, task: str, classes: str | None = None) -> None:
        # Determine the task based on the target or use the provided task
        if task is None:
            target_to_task_map = {
                targets.IRMSD: targets.REGRESS,
                targets.LRMSD: targets.REGRESS,
                targets.FNAT: targets.REGRESS,
                targets.DOCKQ: targets.REGRESS,
                targets.BINARY: targets.CLASSIF,
                targets.CAPRI: targets.CLASSIF,
            }
            self.task = target_to_task_map.get(self.target)
        else:
            self.task = task

        # Validate the task
        if self.task not in [targets.CLASSIF, targets.REGRESS] and self.target is not None:
            msg = f"User target detected: {self.target} -> The task argument must be 'classif' or 'regress', currently set as {self.task}"
            raise ValueError(msg)

        # Warn if the user-set task does not match the determined task
        if task and task != self.task:
            warnings.warn(
                f"Target {self.target} expects {self.task}, but was set to task {task} by user. User set task is ignored and {self.task} will be used.",
            )

        # Handle classification task
        if self.task == targets.CLASSIF:
            if classes is None:
                self.classes = [0, 1, 2, 3, 4, 5] if self.target == targets.CAPRI else [0, 1]
            self.classes_to_index = {class_: index for index, class_ in enumerate(self.classes)}
            _log.info(f"Target classes set to: {self.classes}")
        else:
            self.classes = None
            self.classes_to_index = None

    def _check_inherited_params(
        self,
        inherited_params: list[str],
        data: dict | GraphDataset | GridDataset,
    ) -> None:
        """Check if the parameters for validation and/or testing are the same as in the pre-trained model or training set provided.

        Args:
            inherited_params: List of parameters that need to be checked for inheritance.
            data: The parameters in `inherited_param` will be inherited from the information contained in `data`.
        """
        self_vars = vars(self)
        if not isinstance(data, dict):
            data = vars(data)

        for param in inherited_params:
            if self_vars[param] != data[param]:
                if self_vars[param] != self.default_vars[param]:
                    _log.warning(
                        f"The {param} parameter set here is: {self_vars[param]}, "
                        f"which is not equivalent to the one in the training phase: {data[param]}./n"
                        f"Overwriting {param} parameter with the one used in the training phase.",
                    )
                setattr(self, param, data[param])

    def _create_index_entries(self) -> None:
        """Creates the indexing of each molecule in the dataset.

        Creates the indexing: [ ('1ak4.hdf5,1AK4_100w),...,('1fqj.hdf5,1FGJ_400w)].
        This allows to refer to one entry with its index in the list.
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

                    # skip self._filter_targets when target_filter is None, improve performance using list comprehension.
                    if self.target_filter is None:
                        self.index_entries += [(hdf5_path, entry_name) for entry_name in entry_names]
                    else:
                        self.index_entries += [(hdf5_path, entry_name) for entry_name in entry_names if self._filter_targets(hdf5_file[entry_name])]

            except Exception:  # noqa: BLE001
                _log.exception(f"on {hdf5_path}")

    def _filter_targets(self, grp: h5py.Group) -> bool:
        """Filters the entry according to a dictionary.

        The filter is based on the attribute self.target_filter that must be either
        of the form: { target_name : target_condition } or None.

        Args:
            grp: The entry group in the .HDF5 file.

        Returns:
            True if we keep the entry; False otherwise.

        Raises:
            ValueError: If an unsuported condition is provided.
        """
        if self.target_filter is None:
            return True

        for target_name, target_condition in self.target_filter.items():
            present_target_names = list(grp[targets.VALUES].keys())

            if target_name in present_target_names:
                # If we have a given target_condition, see if it's met.
                if isinstance(target_condition, str):
                    operation = target_condition
                    target_value = grp[targets.VALUES][target_name][()]
                    for operator_string in [">", "<", "==", "<=", ">=", "!="]:
                        operation = operation.replace(operator_string, f"{target_value}" + operator_string)

                    if not eval(operation):  # noqa: S307, PGH001
                        return False

                elif target_condition is not None:
                    msg = "Conditions not supported"
                    raise ValueError(msg, target_condition)

            else:
                _log.warning(f"   :Filter {target_name} not found for entry {grp}\n   :Filter options are: {present_target_names}")
        return True

    def len(self) -> int:
        """Gets the length of the dataset, either :class:`GridDataset` or :class:`GraphDataset` object.

        Returns:
            int: Number of complexes in the dataset.
        """
        return len(self.index_entries)

    def hdf5_to_pandas(  # noqa: C901
        self,
    ) -> pd.DataFrame:
        """Loads features data from the HDF5 files into a Pandas DataFrame in the attribute `df` of the class.

        Returns:
            :class:`pd.DataFrame`: Pandas DataFrame containing the selected features as columns per all data points in
                hdf5_path files.
        """
        df_final = pd.DataFrame()

        for fname in self.hdf5_paths:
            with h5py.File(fname, "r") as f:
                entry_name = next(iter(f.keys()))

                if self.subset is not None:
                    entry_names = [entry for entry, _ in f.items() if entry in self.subset]
                else:
                    entry_names = [entry for entry, _ in f.items()]

                df_dict = {}
                df_dict["id"] = entry_names

                for feat_type in self.features_dict:
                    for feat in self.features_dict[feat_type]:
                        # reset transform for each feature
                        transform = None
                        if self.features_transform:
                            transform = self.features_transform.get("all", {}).get("transform")
                            if (transform is None) and (feat in self.features_transform):
                                transform = self.features_transform.get(feat, {}).get("transform")
                        # Check the number of channels the features have
                        if f[entry_name][feat_type][feat][()].ndim == 2:  # noqa:PLR2004
                            for i in range(f[entry_name][feat_type][feat][:].shape[1]):
                                df_dict[feat + "_" + str(i)] = [f[entry_name][feat_type][feat][:][:, i] for entry_name in entry_names]
                                # apply transformation for each channel in this feature
                                if transform:
                                    df_dict[feat + "_" + str(i)] = [transform(row) for row in df_dict[feat + "_" + str(i)]]
                        else:
                            df_dict[feat] = [
                                f[entry_name][feat_type][feat][:] if f[entry_name][feat_type][feat][()].ndim == 1 else f[entry_name][feat_type][feat][()]
                                for entry_name in entry_names
                            ]
                            # apply transformation
                            if transform:
                                df_dict[feat] = [transform(row) for row in df_dict[feat]]

                df_temp = pd.DataFrame(data=df_dict)
            df_concat = pd.concat([df_final, df_temp])
        self.df = df_concat.reset_index(drop=True)
        return self.df

    def save_hist(  # noqa: C901
        self,
        features: str | list[str],
        fname: str = "features_hist.png",
        bins: int | list[float] | str = 10,
        figsize: tuple = (15, 15),
        log: bool = False,
    ) -> None:
        """After having generated a pd.DataFrame using hdf5_to_pandas method, histograms of the features can be saved in an image.

        Args:
            features: Features to be plotted.
            fname: str or path-like or binary file-like object. Defaults to 'features_hist.png'.
            bins: If bins is an integer, it defines the number of equal-width bins in the range.
                If bins is a sequence, it defines the bin edges, including the left edge of the first bin and the right edge
                of the last bin; in this case, bins may be unequally spaced. All but the last (righthand-most) bin is half-open.
                If bins is a string, it is one of the binning strategies supported by numpy.histogram_bin_edges:
                'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
                Defaults to 10.
            figsize: Saved figure sizes. Defaults to (15, 15).
            log: Whether to apply log transformation to the data indicated by the `features` parameter. Defaults to False.
        """
        if self.df is None:
            self.hdf5_to_pandas()

        if not isinstance(features, list):
            features = [features]

        features_df = [col for feat in features for col in self.df.columns.to_numpy().tolist() if feat in col]

        means = [
            round(np.concatenate(self.df[feat].to_numpy()).mean(), 1)
            if isinstance(self.df[feat].to_numpy()[0], np.ndarray)
            else round(self.df[feat].to_numpy().mean(), 1)
            for feat in features_df
        ]
        devs = [
            round(np.concatenate(self.df[feat].to_numpy()).std(), 1)
            if isinstance(self.df[feat].to_numpy()[0], np.ndarray)
            else round(self.df[feat].to_numpy().std(), 1)
            for feat in features_df
        ]

        if len(features_df) > 1:
            fig, axs = plt.subplots(len(features_df), figsize=figsize)

            for row, feat in enumerate(features_df):
                if isinstance(self.df[feat].to_numpy()[0], np.ndarray):
                    if log:
                        log_data = np.log(np.concatenate(self.df[feat].to_numpy()))
                        log_data[log_data == -np.inf] = 0
                        axs[row].hist(log_data, bins=bins)
                    else:
                        axs[row].hist(np.concatenate(self.df[feat].to_numpy()), bins=bins)
                elif log:
                    log_data = np.log(self.df[feat].to_numpy())
                    log_data[log_data == -np.inf] = 0
                    axs[row].hist(log_data, bins=bins)
                else:
                    axs[row].hist(self.df[feat].to_numpy(), bins=bins)
                axs[row].set(
                    xlabel=f"{feat} (mean {means[row]}, std {devs[row]})",
                    ylabel="Count",
                )
            fig.tight_layout()

        elif len(features_df) == 1:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            if isinstance(self.df[features_df[0]].to_numpy()[0], np.ndarray):
                if log:
                    log_data = np.log(np.concatenate(self.df[features_df[0]].to_numpy()))
                    log_data[log_data == -np.inf] = 0
                    ax.hist(log_data, bins=bins)
                else:
                    ax.hist(np.concatenate(self.df[features_df[0]].to_numpy()), bins=bins)
            elif log:
                log_data = np.log(self.df[features_df[0]].to_numpy())
                log_data[log_data == -np.inf] = 0
                ax.hist(log_data, bins=bins)
            else:
                ax.hist(self.df[features_df[0]].values, bins=bins)
            ax.set(
                xlabel=f"{features_df[0]} (mean {means[0]}, std {devs[0]})",
                ylabel="Count",
            )

        else:
            msg = "Please provide valid features names. They must be present in the current :class:`DeeprankDataset` children instance."
            raise ValueError(msg)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout()
            fig.savefig(fname)
            plt.close(fig)

    def _compute_mean_std(self) -> None:
        means = {
            col: round(np.nanmean(np.concatenate(self.df[col].values)), 1)
            if isinstance(self.df[col].to_numpy()[0], np.ndarray)
            else round(np.nanmean(self.df[col].to_numpy()), 1)
            for col in self.df.columns[1:]
        }
        devs = {
            col: round(np.nanstd(np.concatenate(self.df[col].to_numpy())), 1)
            if isinstance(self.df[col].to_numpy()[0], np.ndarray)
            else round(np.nanstd(self.df[col].to_numpy()), 1)
            for col in self.df.columns[1:]
        }
        self.means = means
        self.devs = devs


# Grid features are stored per dimension and named accordingly.
# Example: position_001, position_002, position_003 (for x,y,z)
# Use this regular expression to take the feature name apart
GRID_PARTIAL_FEATURE_NAME_PATTERN = re.compile(r"^([a-zA-Z_]+)_([0-9]{3})$")


class GridDataset(DeeprankDataset):
    """Class to load the .HDF5 files data into grids.

    Args:
        hdf5_path: Path to .HDF5 file(s). For multiple .HDF5 files, insert the paths in a list. Defaults to None.
        subset: list of keys from .HDF5 file to include. Defaults to None (meaning include all).
        train_source: data to inherit information from the training dataset or the pre-trained model.
            If None, the current dataset is considered as the training set. Otherwise, `train_source` needs to be a dataset of the same class or
            the path of a DeepRank2 pre-trained model. If set, the parameters `features`, `target`, `traget_transform`, `task`, and `classes`
            will be inherited from `train_source`.
            Defaults to None.
        features: Consider all pre-computed features ("all") or some defined node features
            (provide a list, example: ["res_type", "polarity", "bsa"]). The complete list can be found in `deeprank2.domain.gridstorage`.
            Value will be ignored and inherited from `train_source` if `train_source` is assigned.
            Defaults to "all".
        target: Default options are irmsd, lrmsd, fnat, binary, capri_class, and dockq. It can also be
            a custom-defined target given to the Query class as input (see: `deeprank2.query`); in this case,
            the task parameter needs to be explicitly specified as well.
            Only numerical target variables are supported, not categorical.
            If the latter is your case, please convert the categorical classes into
            numerical class indices before defining the :class:`GraphDataset` instance.
            Value will be ignored and inherited from `train_source` if `train_source` is assigned.
            Defaults to None.
        target_transform: Apply a log and then a sigmoid transformation to the target (for regression only).
            This puts the target value between 0 and 1, and can result in a more uniform target distribution and speed up the optimization.
            Value will be ignored and inherited from `train_source` if `train_source` is assigned.
            Defaults to False.
        target_filter: Dictionary of type [target: cond] to filter the molecules.
            Note that the you can filter on a different target than the one selected as the dataset target.
            Defaults to None.
        task: 'regress' for regression or 'classif' for classification. Required if target not in
            ['irmsd', 'lrmsd', 'fnat', 'binary', 'capri_class', or 'dockq'], otherwise this setting is ignored.
            Automatically set to 'classif' if the target is 'binary' or 'capri_classes'.
            Automatically set to 'regress' if the target is 'irmsd', 'lrmsd', 'fnat', or 'dockq'.
            Value will be ignored and inherited from `train_source` if `train_source` is assigned.
            Defaults to None.
        classes: Define the dataset target classes in classification mode.
            Value will be ignored and inherited from `train_source` if `train_source` is assigned.
            Defaults to None.
        use_tqdm: Show progress bar. Defaults to True.
        root: Root directory where the dataset should be saved. Defaults to "./".
        check_integrity: Whether to check the integrity of the hdf5 files. Defaults to True.
    """

    def __init__(
        self,
        hdf5_path: str | list,
        subset: list[str] | None = None,
        train_source: str | GridDataset | None = None,
        features: list[str] | str | None = "all",
        target: str | None = None,
        target_transform: bool = False,
        target_filter: dict[str, str] | None = None,
        task: Literal["regress", "classif"] | None = None,
        classes: list[str] | list[int] | list[float] | None = None,
        use_tqdm: bool = True,
        root: str = "./",
        check_integrity: bool = True,
    ):
        super().__init__(
            hdf5_path,
            subset,
            train_source,
            target,
            target_transform,
            target_filter,
            task,
            classes,
            use_tqdm,
            root,
            check_integrity,
        )
        self.default_vars = {k: v.default for k, v in inspect.signature(self.__init__).parameters.items() if v.default is not inspect.Parameter.empty}
        self.default_vars["classes_to_index"] = None
        self.features = features
        self.target_transform = target_transform

        if train_source is not None:
            self.inherited_params = [
                "features",
                "target",
                "target_transform",
                "task",
                "classes",
                "classes_to_index",
            ]
            self._check_and_inherit_train(GridDataset, self.inherited_params)
            self._check_features()

        else:
            self._check_features()
            self.inherited_params = None

            try:
                fname, mol = self.index_entries[0]
            except IndexError as e:
                msg = "No entries found in the dataset. Please check the dataset parameters."
                raise IndexError(msg) from e
            with h5py.File(fname, "r") as f5:
                grp = f5[mol]
                possible_targets = grp[targets.VALUES].keys()
                if self.target is None:
                    msg = f"Please set the target during training dataset definition; targets present in the file/s are {possible_targets}."
                    raise ValueError(msg)
                if self.target not in possible_targets:
                    msg = f"Target {self.target} not present in the file/s; targets present in the file/s are {possible_targets}."
                    raise ValueError(msg)

        self.features_dict = {}
        self.features_dict[gridstorage.MAPPED_FEATURES] = self.features
        if self.target is not None:
            if isinstance(self.target, str):
                self.features_dict[targets.VALUES] = [self.target]
            else:
                self.features_dict[targets.VALUES] = self.target

    def _check_features(self) -> None:  # noqa: C901
        """Checks if the required features exist."""
        hdf5_path = self.hdf5_paths[0]

        # read available features
        with h5py.File(hdf5_path, "r") as f:
            mol_key = next(iter(f.keys()))
            if isinstance(self.features, list):
                self.features = [
                    GRID_PARTIAL_FEATURE_NAME_PATTERN.match(feature_name).group(1)
                    if GRID_PARTIAL_FEATURE_NAME_PATTERN.match(feature_name) is not None
                    else feature_name
                    for feature_name in self.features
                ]  # remove the dimension number suffix
                self.features = list(set(self.features))  # remove duplicates
            available_features = list(f[f"{mol_key}/{gridstorage.MAPPED_FEATURES}"].keys())
            available_features = [key for key in available_features if key[0] != "_"]  # ignore metafeatures

            hdf5_matching_feature_names = []  # feature names that match with the requested list of names
            unpartial_feature_names = []  # feature names without their dimension number suffix
            for feature_name in available_features:
                partial_feature_match = GRID_PARTIAL_FEATURE_NAME_PATTERN.match(feature_name)
                if partial_feature_match is not None:  # there's a dimension number in the feature name
                    unpartial_feature_name = partial_feature_match.group(1)

                    if self.features == "all" or (isinstance(self.features, list) and unpartial_feature_name in self.features):
                        hdf5_matching_feature_names.append(feature_name)

                    unpartial_feature_names.append(unpartial_feature_name)

                else:  # no numbers, it's a one-dimensional feature name
                    if self.features == "all" or (isinstance(self.features, list) and feature_name in self.features):
                        hdf5_matching_feature_names.append(feature_name)

                    unpartial_feature_names.append(feature_name)

        # check for the requested features
        missing_features = []
        if self.features == "all":
            self.features = sorted(available_features)
            self.default_vars["features"] = self.features
        else:
            if not isinstance(self.features, list):
                if self.features is None:
                    self.features = []
                else:
                    self.features = [self.features]
            for feature_name in self.features:
                if feature_name not in unpartial_feature_names:
                    _log.info(f"The feature {feature_name} was not found in the file {hdf5_path}.")
                    missing_features.append(feature_name)

            self.features = sorted(hdf5_matching_feature_names)

        # raise error if any features are missing
        if len(missing_features) > 0:
            msg = (
                f"Not all features could be found in the file {hdf5_path} under entry {mol_key}.\n\t"
                f"Missing features are: {missing_features}.\n\t"
                "Check feature_modules passed to the preprocess function.\n\t"
                "Probably, the feature wasn't generated during the preprocessing step.\n\t"
                f"Available features: {available_features}"
            )
            raise ValueError(msg)

    def get(self, idx: int) -> Data:
        """Gets one grid item from its unique index.

        Args:
            idx: Index of the item, ranging from 0 to len(dataset).

        Returns:
            :class:`torch_geometric.data.data.Data`: item with tensors x, y if present, entry_names.
        """
        file_path, entry_name = self.index_entries[idx]
        return self.load_one_grid(file_path, entry_name)

    def load_one_grid(self, hdf5_path: str, entry_name: str) -> Data:
        """Loads one grid.

        Args:
            hdf5_path: .HDF5 file name.
            entry_name: Name of the entry.

        Returns:
            :class:`torch_geometric.data.data.Data`: item with tensors x, y if present, entry_names.
        """
        with h5py.File(hdf5_path, "r") as hdf5_file:
            grp = hdf5_file[entry_name]

            mapped_features_group = grp[gridstorage.MAPPED_FEATURES]

            feature_data = [mapped_features_group[feature_name][:] for feature_name in self.features if feature_name[0] != "_"]
            x = torch.tensor(np.expand_dims(np.array(feature_data), axis=0), dtype=torch.float)

            # target
            if self.target is None:
                y = None
            elif targets.VALUES in grp and self.target in grp[targets.VALUES]:
                y = torch.tensor([grp[targets.VALUES][self.target][()]], dtype=torch.float)

                if self.task == targets.REGRESS and self.target_transform is True:
                    y = torch.sigmoid(torch.log(y))
                elif self.task is not targets.REGRESS and self.target_transform is True:
                    msg = f'Sigmoid transformation not possible for {self.task} tasks. Please change `task` to "regress" or set `target_transform` to `False`.'
                    raise ValueError(msg)
            else:
                y = None
                possible_targets = grp[targets.VALUES].keys()
                if self.train_source is None:
                    msg = (
                        f"Target {self.target} missing in entry {entry_name} in file {hdf5_path}, possible targets are {possible_targets}.\n\t"
                        "Use the query class to add more target values to input data."
                    )
                    raise ValueError(msg)

        # Wrap up the data in this object, for the collate_fn to handle it properly:
        data = Data(x=x, y=y)
        data.entry_names = entry_name

        return data


class GraphDataset(DeeprankDataset):
    """Class to load the .HDF5 files data into graphs.

    Args:
        hdf5_path: Path to .HDF5 file(s). For multiple .HDF5 files, insert the paths in a list. Defaults to None.
        subset: list of keys from .HDF5 file to include. Defaults to None (meaning include all).
        train_source: data to inherit information from the training dataset or the pre-trained model.
            If None, the current dataset is considered as the training set. Otherwise, `train_source` needs to be a dataset of the same class or
            the path of a DeepRank2 pre-trained model. If set, the parameters `node_features`, `edge_features`, `features_transform`,
            `target`, `target_transform`, `task`, and `classes` will be inherited from `train_source`. If standardization was performed in the
            training dataset/step, also the attributes `means` and `devs` will be inherited from `train_source`, and they will be used to scale
            the validation/testing set.
            Defaults to None.
        node_features: Consider all pre-computed node features ("all") or some defined node features (provide a list, e.g.: ["res_type", "polarity", "bsa"]).
            The complete list can be found in `deeprank2.domain.nodestorage`.
            Value will be ignored and inherited from `train_source` if `train_source` is assigned.
            Defaults to "all".
        edge_features: Consider all pre-computed edge features ("all") or some defined edge features (provide a list, e.g.: ["dist", "coulomb"]).
            The complete list can be found in `deeprank2.domain.edgestorage`.
            Value will be ignored and inherited from `train_source` if `train_source` is assigned.
            Defaults to "all".
        features_transform: Dictionary to indicate the transformations to apply to each feature in the dictionary, being the
            transformations lambda functions and/or standardization.
            Example: `features_transform = {'bsa': {'transform': lambda t:np.log(t+1),' standardize': True}}` for the feature `bsa`.
            An `all` key can be set in the dictionary for indicating to apply the same `standardize` and `transform` to all the features.
            Example: `features_transform = {'all': {'transform': lambda t:np.log(t+1), 'standardize': True}}`.
            If both `all` and feature name/s are present, the latter have the priority over what indicated in `all`.
            Value will be ignored and inherited from `train_source` if `train_source` is assigned.
            Defaults to None.
        clustering_method: "mcl" for Markov cluster algorithm (see https://micans.org/mcl/), or "louvain" for Louvain method (see https://en.wikipedia.org/wiki/Louvain_method).
            In both options, for each graph, the chosen method first finds communities (clusters) of nodes and generates
            a torch tensor whose elements represent the cluster to which the node belongs to. Each tensor is then saved
            in the .HDF5 file as a :class:`Dataset` called "depth_0". Then, all cluster members beloging to the same
            community are pooled into a single node, and the resulting tensor is used to find communities among the
            pooled clusters. The latter tensor is saved into the .HDF5 file as a :class:`Dataset` called "depth_1". Both
            "depth_0" and "depth_1" :class:`Datasets` belong to the "cluster" Group. They are saved in the .HDF5 file to
            make them available to networks that make use of clustering methods. Defaults to None.
        target: Default options are irmsd, lrmsd, fnat, binary, capri_class, and dockq.
            It can also be a custom-defined target given to the Query class as input (see: `deeprank2.query`);
            in this case, the task parameter needs to be explicitly specified as well.
            Only numerical target variables are supported, not categorical.
            If the latter is your case, please convert the categorical classes into
            numerical class indices before defining the :class:`GraphDataset` instance.
            Value will be ignored and inherited from `train_source` if `train_source` is assigned.
            Defaults to None.
        target_transform: Apply a log and then a sigmoid transformation to the target (for regression only).
            This puts the target value between 0 and 1, and can result in a more uniform target distribution and speed up the optimization.
            Value will be ignored and inherited from `train_source` if `train_source` is assigned.
            Defaults to False.
        target_filter: Dictionary of type [target: cond] to filter the molecules.
            Note that the you can filter on a different target than the one selected as the dataset target.
            Defaults to None.
        task: 'regress' for regression or 'classif' for classification. Required if target not in
            ['irmsd', 'lrmsd', 'fnat', 'binary', 'capri_class', or 'dockq'], otherwise this setting is ignored.
            Automatically set to 'classif' if the target is 'binary' or 'capri_classes'.
            Automatically set to 'regress' if the target is 'irmsd', 'lrmsd', 'fnat', or 'dockq'.
            Value will be ignored and inherited from `train_source` if `train_source` is assigned.
            Defaults to None.
        classes: Define the dataset target classes in classification mode.
            Value will be ignored and inherited from `train_source` if `train_source` is assigned.
            Defaults to None.
        use_tqdm: Show progress bar. Defaults to True.
        root: Root directory where the dataset should be saved. Defaults to "./".
        check_integrity: Whether to check the integrity of the hdf5 files. Defaults to True.
    """

    def __init__(  # noqa: C901
        self,
        hdf5_path: str | list,
        subset: list[str] | None = None,
        train_source: str | GridDataset | None = None,
        node_features: list[str] | str | None = "all",
        edge_features: list[str] | str | None = "all",
        features_transform: dict | None = None,
        clustering_method: str | None = None,
        target: str | None = None,
        target_transform: bool = False,
        target_filter: dict[str, str] | None = None,
        task: Literal["regress", "classif"] | None = None,
        classes: list[str] | list[int] | list[float] | None = None,
        use_tqdm: bool = True,
        root: str = "./",
        check_integrity: bool = True,
    ):
        super().__init__(
            hdf5_path,
            subset,
            train_source,
            target,
            target_transform,
            target_filter,
            task,
            classes,
            use_tqdm,
            root,
            check_integrity,
        )

        self.default_vars = {k: v.default for k, v in inspect.signature(self.__init__).parameters.items() if v.default is not inspect.Parameter.empty}
        self.default_vars["classes_to_index"] = None
        self.node_features = node_features
        self.edge_features = edge_features
        self.clustering_method = clustering_method
        self.target_transform = target_transform
        self.features_transform = features_transform

        if train_source is not None:
            self.inherited_params = [
                "node_features",
                "edge_features",
                "features_transform",
                "target",
                "target_transform",
                "task",
                "classes",
                "classes_to_index",
            ]
            self._check_and_inherit_train(GraphDataset, self.inherited_params)
            self._check_features()

        else:
            self._check_features()
            self.inherited_params = None

            try:
                fname, mol = self.index_entries[0]
            except IndexError as e:
                msg = "No entries found in the dataset. Please check the dataset parameters."
                raise IndexError(msg) from e
            with h5py.File(fname, "r") as f5:
                grp = f5[mol]
                possible_targets = grp[targets.VALUES].keys()
                if self.target is None:
                    msg = f"Please set the target during training dataset definition; targets present in the file/s are {possible_targets}."
                    raise ValueError(msg)
                if self.target not in possible_targets:
                    msg = f"Target {self.target} not present in the file/s; targets present in the file/s are {possible_targets}."
                    raise ValueError(msg)

        self.features_dict = {}
        self.features_dict[Nfeat.NODE] = self.node_features
        self.features_dict[Efeat.EDGE] = self.edge_features
        if self.target is not None:
            if isinstance(self.target, str):
                self.features_dict[targets.VALUES] = [self.target]
            else:
                self.features_dict[targets.VALUES] = self.target

        standardize = False
        if self.features_transform:
            standardize = any(self.features_transform[key].get("standardize") for key, _ in self.features_transform.items())

        if standardize and (train_source is None):
            if self.means or self.devs is None:
                if self.df is None:
                    self.hdf5_to_pandas()
                self._compute_mean_std()
        elif standardize and (train_source is not None):
            self.means = self.train_means
            self.devs = self.train_devs

    def get(self, idx: int) -> Data:
        """Gets one graph item from its unique index.

        Args:
            idx: Index of the item, ranging from 0 to len(dataset).

        Returns:
            :class:`torch_geometric.data.data.Data`: item with tensors x, y if present, edge_index, edge_attr, pos, entry_names.
        """
        fname, mol = self.index_entries[idx]
        return self.load_one_graph(fname, mol)

    def load_one_graph(self, fname: str, entry_name: str) -> Data:  # noqa: PLR0915, C901
        """Loads one graph.

        Args:
            fname: .HDF5 file name.
            entry_name: Name of the entry.

        Returns:
            :class:`torch_geometric.data.data.Data`: item with tensors x, y if present, edge_index, edge_attr, pos, entry_names.
        """
        with h5py.File(fname, "r") as f5:
            grp = f5[entry_name]

            # node features
            if len(self.node_features) > 0:
                node_data = ()
                for feat in self.node_features:
                    # resetting transformation and standardization for each feature
                    transform = None
                    standard = None

                    if feat[0] != "_":  # ignore metafeatures
                        vals = grp[f"{Nfeat.NODE}/{feat}"][()]
                        # get feat transformation and standardization
                        if self.features_transform is not None:
                            transform = self.features_transform.get("all", {}).get("transform")
                            standard = self.features_transform.get("all", {}).get("standardize")
                            # if no transformation is set for all features, check if one is set for the current feature
                            if (transform is None) and (feat in self.features_transform):
                                transform = self.features_transform.get(feat, {}).get("transform")
                            # if no standardization is set for all features, check if one is set for the current feature
                            if (standard is None) and (feat in self.features_transform):
                                standard = self.features_transform.get(feat, {}).get("standardize")

                        # apply transformation
                        if transform:
                            with warnings.catch_warnings(record=True) as w:
                                vals = transform(vals)
                                if len(w) > 0:
                                    msg = (
                                        f"Invalid value occurs in {entry_name}, file {fname},when applying {transform} for feature {feat}.\n\t"
                                        f"Please change the transformation function for {feat}."
                                    )
                                    raise ValueError(msg)

                        if vals.ndim == 1:  # features with only one channel
                            vals = vals.reshape(-1, 1)
                            if standard:
                                vals = (vals - self.means[feat]) / self.devs[feat]
                        elif standard:
                            reshaped_mean = [mean_value for mean_key, mean_value in self.means.items() if feat in mean_key]
                            reshaped_dev = [dev_value for dev_key, dev_value in self.devs.items() if feat in dev_key]
                            vals = (vals - reshaped_mean) / reshaped_dev
                        node_data += (vals,)
                x = torch.tensor(np.hstack(node_data), dtype=torch.float)
            else:
                x = None
                _log.warning("No node features set.")

            # edge index,
            # we have to have all the edges i.e : (i,j) and (j,i)
            if Efeat.INDEX in grp[Efeat.EDGE]:
                ind = grp[f"{Efeat.EDGE}/{Efeat.INDEX}"][()]
                if ind.ndim == 2:  # noqa: PLR2004
                    ind = np.vstack((ind, np.flip(ind, 1))).T
                edge_index = torch.tensor(ind, dtype=torch.long).contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            # edge feature
            # we have to have all the edges i.e : (i,j) and (j,i)
            if len(self.edge_features) > 0:
                edge_data = ()
                for feat in self.edge_features:
                    # resetting transformation and standardization for each feature
                    transform = None
                    standard = None

                    if feat[0] != "_":  # ignore metafeatures
                        vals = grp[f"{Efeat.EDGE}/{feat}"][()]
                        # get feat transformation and standardization
                        if self.features_transform is not None:
                            transform = self.features_transform.get("all", {}).get("transform")
                            standard = self.features_transform.get("all", {}).get("standardize")
                            # if no transformation is set for all features, check if one is set for the current feature
                            if (transform is None) and (feat in self.features_transform):
                                transform = self.features_transform.get(feat, {}).get("transform")
                            # if no standardization is set for all features, check if one is set for the current feature
                            if (standard is None) and (feat in self.features_transform):
                                standard = self.features_transform.get(feat, {}).get("standardize")

                        # apply transformation
                        if transform:
                            with warnings.catch_warnings(record=True) as w:
                                vals = transform(vals)
                                if len(w) > 0:
                                    msg = (
                                        f"Invalid value occurs in {entry_name}, file {fname}, when applying {transform} for feature {feat}.\n\t"
                                        f"Please change the transformation function for {feat}."
                                    )
                                    raise ValueError(msg)

                        if vals.ndim == 1:
                            vals = vals.reshape(-1, 1)
                            if standard:
                                vals = (vals - self.means[feat]) / self.devs[feat]
                        elif standard:
                            reshaped_mean = [mean_value for mean_key, mean_value in self.means.items() if feat in mean_key]
                            reshaped_dev = [dev_value for dev_key, dev_value in self.devs.items() if feat in dev_key]
                            vals = (vals - reshaped_mean) / reshaped_dev
                        edge_data += (vals,)
                edge_data = np.hstack(edge_data)
                edge_data = np.vstack((edge_data, edge_data))
                edge_attr = torch.tensor(edge_data, dtype=torch.float).contiguous()
            else:
                edge_attr = torch.empty((edge_index.shape[1], 0), dtype=torch.float).contiguous()

            # target
            if self.target is None:
                y = None
            elif targets.VALUES in grp and self.target in grp[targets.VALUES]:
                y = torch.tensor([grp[f"{targets.VALUES}/{self.target}"][()]], dtype=torch.float).contiguous()

                if self.task == targets.REGRESS and self.target_transform is True:
                    y = torch.sigmoid(torch.log(y))
                elif self.task is not targets.REGRESS and self.target_transform is True:
                    msg = f'Sigmoid transformation not possible for {self.task} tasks. Please change `task` to "regress" or set `target_transform` to `False`.'
                    raise ValueError(msg)

            else:
                y = None
                possible_targets = grp[targets.VALUES].keys()
                if self.train_source is None:
                    msg = (
                        f"Target {self.target} missing in entry {entry_name} in file {fname}, possible targets are {possible_targets}.\n\t"
                        "Use the query class to add more target values to input data."
                    )
                    raise ValueError(msg)

            # positions
            pos = torch.tensor(grp[f"{Nfeat.NODE}/{Nfeat.POSITION}/"][()], dtype=torch.float).contiguous()

            # cluster
            cluster0 = None
            cluster1 = None
            if self.clustering_method is not None and "clustering" in grp:
                if self.clustering_method in grp["clustering"]:
                    if "depth_0" in grp[f"clustering/{self.clustering_method}"] and "depth_1" in grp[f"clustering/{self.clustering_method}"]:
                        cluster0 = torch.tensor(
                            grp["clustering/" + self.clustering_method + "/depth_0"][()],
                            dtype=torch.long,
                        )
                        cluster1 = torch.tensor(
                            grp["clustering/" + self.clustering_method + "/depth_1"][()],
                            dtype=torch.long,
                        )
                    else:
                        _log.warning("no clusters detected")
                else:
                    _log.warning(f"no clustering/{self.clustering_method} detected")

        # load
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)

        data.cluster0 = cluster0
        data.cluster1 = cluster1

        data.entry_names = entry_name

        return data

    def _check_features(self) -> None:  # noqa: C901
        """Checks if the required features exist."""
        f = h5py.File(self.hdf5_paths[0], "r")
        mol_key = next(iter(f.keys()))

        # read available node features
        self.available_node_features = list(f[f"{mol_key}/{Nfeat.NODE}/"].keys())
        self.available_node_features = [key for key in self.available_node_features if key[0] != "_"]  # ignore metafeatures

        # read available edge features
        self.available_edge_features = list(f[f"{mol_key}/{Efeat.EDGE}/"].keys())
        self.available_edge_features = [key for key in self.available_edge_features if key[0] != "_"]  # ignore metafeatures

        f.close()

        # check node features
        missing_node_features = []
        if self.node_features == "all":
            self.node_features = self.available_node_features
            self.default_vars["node_features"] = self.node_features
        else:
            if not isinstance(self.node_features, list):
                if self.node_features is None:
                    self.node_features = []
                else:
                    self.node_features = [self.node_features]
            for feat in self.node_features:
                if feat not in self.available_node_features:
                    _log.info(f"The node feature _{feat}_ was not found in the file {self.hdf5_paths[0]}.")
                    missing_node_features.append(feat)

        # check edge features
        missing_edge_features = []
        if self.edge_features == "all":
            self.edge_features = self.available_edge_features
            self.default_vars["edge_features"] = self.edge_features
        else:
            if not isinstance(self.edge_features, list):
                if self.edge_features is None:
                    self.edge_features = []
                else:
                    self.edge_features = [self.edge_features]
            for feat in self.edge_features:
                if feat not in self.available_edge_features:
                    _log.info(f"The edge feature _{feat}_ was not found in the file {self.hdf5_paths[0]}.")
                    missing_edge_features.append(feat)

        # raise error if any features are missing
        if missing_node_features + missing_edge_features:
            miss_node_error, miss_edge_error = "", ""
            _log.info(
                "\nCheck feature_modules passed to the preprocess function.\
                Probably, the feature wasn't generated during the preprocessing step.",
            )
            if missing_node_features:
                _log.info(f"\nAvailable node features: {self.available_node_features}\n")
                miss_node_error = f"\nMissing node features: {missing_node_features} \
                                    \nAvailable node features: {self.available_node_features}"
            if missing_edge_features:
                _log.info(f"\nAvailable edge features: {self.available_edge_features}\n")
                miss_edge_error = f"\nMissing edge features: {missing_edge_features} \
                                    \nAvailable edge features: {self.available_edge_features}"
            msg = (
                f"Not all features could be found in the file {self.hdf5_paths[0]}.\n\t"
                "Check feature_modules passed to the preprocess function.\n\t"
                "Probably, the feature wasn't generated during the preprocessing step.\n\t"
                f"{miss_node_error}{miss_edge_error}"
            )
            raise ValueError(msg)


def save_hdf5_keys(
    f_src_path: str,
    src_ids: list[str],
    f_dest_path: str,
    hardcopy: bool = False,
) -> None:
    """Save references to keys in src_ids in a new .HDF5 file.

    Args:
        f_src_path: The path to the .HDF5 file containing the keys.
        src_ids: Keys to be saved in the new .HDF5 file. It should be a list containing at least one key.
        f_dest_path: The path to the new .HDF5 file.
        hardcopy: If False, the new file contains only references (external links, see :class:`ExternalLink` class from `h5py`)
            to the original .HDF5 file.
            If True, the new file contains a copy of the objects specified in src_ids (see h5py :class:`HardLink` from `h5py`).
            Defaults to False.
    """
    if not all(isinstance(d, str) for d in src_ids):
        msg = "data_ids should be a list containing strings."
        raise TypeError(msg)

    with h5py.File(f_dest_path, "w") as f_dest, h5py.File(f_src_path, "r") as f_src:
        for key in src_ids:
            if hardcopy:
                f_src.copy(f_src[key], f_dest)
            else:
                f_dest[key] = h5py.ExternalLink(f_src_path, "/" + key)
